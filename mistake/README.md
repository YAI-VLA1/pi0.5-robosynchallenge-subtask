# mistake 메타데이터 (pi0.7 방식)

프롬프트에 `Mistake: true` / `Mistake: false` 를 넣어, 실패한 궤적을 "실패라고
라벨이 붙은 채로" 학습시킨다. 추론 때는 항상 `false` 를 넣어 "실수 없는 쪽"을 요청한다.

```
baseline  Task: {task}, State: {14개 정수}, Mistake: false;\nAction:
subtask   State: {14개 정수}; Mistake: false; Subtask: {14토큰 슬롯};\nAction:
드롭(5%)  State: {14개 정수}; Subtask: {14토큰 슬롯};\nAction:        <- 기존과 동일
```

**드롭된 형태가 기존 프롬프트와 바이트 단위로 같다.** 기존 체크포인트에서
이어붙일 때 5%는 모델이 알던 입력 그대로 들어간다. pi0.7 도 메타데이터 필드마다
5% 로 드롭한다.

`Mistake` 는 `Subtask` **앞**에 와야 한다 — 슬롯이 causal 이라 생성 시점에
이미 보여야 한다.

---

## 실수 시점(`t_fail`)을 어떻게 정했나

`rollout/stage_metrics.py` 의 `t_fail` 은 이 모델에 **쓸 수 없었다.** 100개 전부
`50` 으로 나온다 — subtask 모델은 굳지 않아서 `t_frozen` 이 전부 null 이고,
`t_stop=0 + FAIL_OFFSET 50` 으로 고정된 값이기 때문이다.

그리퍼 규약을 대본 데모에서 먼저 확인했다: **1 = 열림, 0 = 닫힘**
(subtask 0 `open both grippers` 구간에서 `action[13]` 이 0 → 1 로 간다).

그 위에서 에피소드를 넷으로 나눈다:

| | 수 | `t_fail` |
|---|---|---|
| **A** 잡기 실패 — 그리퍼가 빈 허공에서 닫힘 | 92 | 그리퍼가 닫힌 프레임 |
| **B** 잡았다 놓침 — 들었다가 다시 떨어짐 | 5 | 펜이 다시 내려앉은 프레임 |
| **C** 들고 미완 — 떨어뜨리지도 완수하지도 않음 | 2 | 특정 불가 → **제외** |
| **D** 성공 | 1 | 없음. 안 자르고 전부 `false` |

"들었다"는 판정에 **1초(25프레임) 지속**을 요구한다. 그래야 쳐서 튄 것(ep63: 2프레임,
ep88: 14프레임)과 실제로 집어 든 것을 가른다.

## 라벨과 절단

```
mistake = True   for t >= t_fail - 50      # 그 프레임이 내놓는 50-chunk 가 실수를 포함
절단            [0, t_fail + 50]           # 마지막 mistake 프레임의 action horizon 을 채운다
```

`50` 은 `action_horizon` 이다. 프레임 `t` 가 내놓는 청크가 `[t, t+49]` 를 덮으므로,
`t >= t_fail-50` 인 프레임의 청크만 실수 순간을 포함한다.

절단을 `t_fail + 50` 에서 하는 이유: 거기서 더 자르면 마지막 mistake 프레임의
action 타깃이 패딩으로 채워져 **"패딩을 내라"를 가르치게 된다.**

실측 결과:

```
t_fail        중앙값 86   (최소 77, 최대 234)
남길 프레임     중앙값 137  (원본 349 -> 꼬리 59% 버림)
mistake 프레임  101        (= 50 + 1 + 50)
```

### 덤으로 나온 진단

`t_fail` 중앙값 **86** 인데, 대본의 `close the right gripper on the pen` 구간이
**74~89** 다. 모델은 **정확히 제 시각에 그리퍼를 닫는다.** 타이밍은 맞고 위치만
틀렸다 — 조준 오차다.

---

## 병합 데이터셋

`cobotmagic_Sim_items_handover_mix` = 데모 1,000 + 롤아웃 98

- 데모 parquet·영상, 롤아웃 영상은 전부 **심볼릭 링크** (복사 없음)
- 롤아웃 parquet 98개만 새로 쓴다 — `[0, t_end]` 로 자르고 `episode_index` /
  `frame_index` / `index` 를 다시 매긴다
- **영상은 안 자른다.** LeRobot 은 parquet 길이만큼만 프레임을 요청하므로 뒤에
  남은 프레임은 읽히지 않는다 (AV1 재인코딩은 libaom 이라 몇 시간 걸린다)
- `task` 문자열은 데모와 같은 하나다. 같은 과제를 정책이 실행한 것이고,
  `prompt_from_task` 가 같은 프롬프트를 내야 한다
- `meta/stats.json` 은 데모 것을 그대로 쓴다. 롤아웃이 4% 라 통계는 사실상 같고,
  기존 체크포인트가 본 입력 분포를 안 바꾸는 편이 재개에 안전하다

```
1,098 에피소드 · 341,252 프레임 · mistake=true 9,797 (2.9%)
```

---

## 실행

```bash
export PI05=<repo>/policy/pi05
export ROLLOUTS=/path/to/rollouts/subtask     # mistake_labels.json 이 여기 생긴다

python3 mistake/mistake_labels.py        $ROLLOUTS
python3 mistake/merge_mistake_ds.py
python3 mistake/prepare_mistake_config.py $PI05
python3 mistake/apply_mistake.py          $PI05

# norm_stats 는 baseline 것을 그대로 쓴다
cp $PI05/assets/pi05_robosyn_items_handover_lora_cos82k/RoboSynChallenge/cobotmagic_Sim_items_handover/norm_stats.json \
   $PI05/assets/pi05_robosyn_items_handover_mistake_cos82k/RoboSynChallenge/cobotmagic_Sim_items_handover_mix/

# 검증 — 토큰을 문자열로 되돌려 6항목
cd $PI05 && env PYTHONPATH=src HF_LEROBOT_HOME=$PI05/training_data \
  PI05_MISTAKE=1 .venv/bin/python <repo>/mistake/mistake_smoke.py
```

## 파일

| | |
|---|---|
| `mistake_labels.py` | `t_fail` 판정 + 절단 지점 → `mistake_labels.json` |
| `merge_mistake_ds.py` | 데모 + 롤아웃 병합 → `_mix` 데이터셋 |
| `prepare_mistake_config.py` | `TrainConfig` 2개 등록 (mistake / subtask+mistake) |
| `apply_mistake.py` | 소스 패치 11곳 (tokenizer · transforms · libero_policy · config) |
| `mistake_smoke.py` | 프롬프트 문자열 6항목 검증 |

## 한계

- 우리 100개 실패 데이터에는 **복구 구간이 없다.** 끝까지 실패한다. pi0.7 은
  "실패 후 사람이 교정한" 데이터를 히스토리와 함께 학습했다. 그래서 여기서
  배울 수 있는 것은 "실수한 궤적처럼 하지 마라" 까지고, "실수했으면 이렇게
  고쳐라" 는 아니다.
- `mistake` 가 **출처와 완전히 상관**된다 (대본=false, 롤아웃=true). 모델이
  "mistake 란 롤아웃처럼 생긴 것"을 배우고 추론에서 `false` 가 "대본처럼 해라"
  로만 작동할 수 있다. 효과를 볼 때 이 가능성을 배제해야 한다.
- 92/97 이 **첫 grasp 에서** 실패한다. 그 시점에는 조건 걸 이전 실수가 없다.
