# pi0.5 post-training — RoboSynChallenge `items_handover`

pi0.5 fine-tuning 성공률이 1/100 인 데서 출발해, 올릴 수 있는 축 세 개를 붙였다.
셋은 서로 독립이고 환경변수로 켜고 끈다.

| | 무엇 | 문서 |
|---|---|---|
| **subtask** | 모델이 하위 지시문을 스스로 생성하고 거기 맞춰 action 을 뽑는다 | 이 문서 |
| **mistake** | 실패 궤적에 pi0.7 식 `Mistake: true/false` 를 붙여 조건화한다 | [`mistake/README.md`](mistake/README.md) |
| **MEM** | SigLIP 에 space-time 분리 어텐션 (새 파라미터 0개) | [`mem/README.md`](mem/README.md) |

> **다른 머신에서 돌리려면 → [`SETUP.md`](SETUP.md)**
> 빈 머신에서 학습까지 복붙으로 가는 안내. GPU 별 배치 가이드와 함정 모음 포함.
> 학습 띄우기 전에 `python3 patches/verify_patches.py <repo> --all` 을 **반드시** 통과시킬 것.

---

## subtask — 왜 이걸 먼저 했나

items_handover 에서 pi0.5 가 **subtask 지시문을 스스로 생성하고 거기에 맞춰 action 을 뽑도록**
만드는 작업. 데이터셋에 subtask 주석이 없다는 문제에서 출발한다.

## 왜

pi0.5 fine-tuning (LoRA, 82k step) 결과가 **1/100** 이었다. 원인은
**평가에서 왼쪽 그리퍼가 한 번도 닫히지 않는 것**이다. 학습 손실은 0.00063 까지 내려갔고,
학습 코드·성공 판정·norm_stats·state 순서·이미지 키 매핑을 전부 배제했다.
남은 설명은 covariate shift — 오른팔이 시연과 다른 자세로 인계하니 왼쪽 그리퍼가
닫기를 배운 장면이 평가에서 나타나지 않는다.

subtask conditioning 이 여기에 들어맞는다. "지금 무엇을 하는 중인가"를 명시적으로 주면
그 판단이 action 회귀에서 분리되고, 정책이 인계가 끝난 것처럼 다음 동작으로 넘어가는 것을
막을 수 있다.

## subtask 라벨은 어디서 오나 — 주석이 없는데

**데이터를 만든 스크립트에서 계산한다.** sim 에피소드는
`configs/<task>/action_config.json` 의 키포즈 DAG 를 그대로 재생한 것이고,
edge 마다 `duration` 이 상수, `sync` 가 선후를 묶는다. 스케줄을 풀면 경계가 스텝 단위로 나온다.

```
python3 schedule/schedule.py <rsc_repo/configs> <dataset_root> schedule_sim.json
```

10개 태스크 중 **9개가 복원한 총 길이와 데이터셋 에피소드 길이가 정확히 일치**한다
(handle_basket 만 마지막 `ending` edge 가 가변).

| task | 스케줄 | 데이터 | 구간 |
|---|---|---|---|
| click_bell | 74 | 74 | 2 |
| table_rearrangement | 200 | 200 | 20 |
| water_pouring | 207 | 207 | 17 |
| manipulate_pipette | 286 | 286 | 11 |
| mixer_operating | 316 | 316 | 15 |
| item_assembly | 318 | 318 | 22 |
| items_handover | 327 | 327 | 16 |
| sample_loading | 372 | 372 | 19 |
| drawer_open_place | 425 | 425 | 17 |
| handle_basket | 335 | 322–335 | 17 |

**구간의 좌/우 scope 를 edge 이름으로 추측하면 안 된다** — 156개 중 49개가 틀린다
(`rclose0` 이 우측, `init_to_aim` 이 우완 기본 scope). `action_config.json` 의 `edge`
딕셔너리 키가 정답이고 `schedule.py` 가 `[start, end, scope]` 로 실어 준다.

검증은 `schedule/verify_objects.py` — parquet 의 `*_pose` 로 물체 운동 개시 시점을
경계와 대조한다. items_handover 에피소드 0 에서 전부 일치했다.

## 먼저 전제를 검증했다 (step 1)

sim 은 타이밍이 고정이라(`left_close0` 은 언제나 202–217) subtask 라벨이 timestep 과
거의 1:1 이다. 그러면 예측기가 학습 분포 안에서만 성립하는 mapping 을 배우고,
평가에서 로봇이 이상한 상태에 빠지면 엉뚱한 subtask 를 뱉을 수 있다.

**시간 입력을 일절 받지 않는** 분류기(이미지 3뷰만)로 확인했다.

```
subtask/phases.py       구간 -> 프레임별 단일 phase 라벨 (16구간 -> 15 phase)
subtask/decode.py       영상 -> (T,112,336,3) uint8, demo/eval 동일 형식
subtask/train_phase.py  ResNet18 phase 분류기
subtask/apply_eval.py   평가 롤아웃 적용 + strip chart
subtask/run_pipeline.sh 디코딩 대기 -> 학습 -> 적용
```

**결과: 전제 통과.**

| | |
|---|---|
| 홀드아웃 demo 정확도 | 96.6% (±1 phase 허용 100.0%) |
| 고정 schedule 과의 일치율 — demo | 0.964 |
| 고정 schedule 과의 일치율 — eval fail 100개 | **0.239** |
| eval fail 에서 마지막 phase(9·12·13·14) 체류 | **0.0%** |

policy 가 실제로 못 한 단계를 예측기가 정확히 거부한다. 최대 도달 phase 는 11 에서 60개,
10 에서 30개다. 자세한 것은 [`results/subtask_classifier.md`](results/subtask_classifier.md).

**한계**: phase 8(`left_close0`)이 fail 의 90% 에서 발화한다. 왼팔이 인수 자세까지는
실제로 가고, phase 7 과 8 은 그리퍼 개도로만 갈린다. **가장 필요한 판별이 최약점이다.**
따라서 기대 이득은 "닫아라"라는 trigger 가 아니라 **"아직 넘어가지 마라"라는 순서 신호** 쪽이다.

## step 2 — 구현 (학습 진행 중)

prompt 에 Subtask 슬롯을 만들고, 그 토큰 위치에만 CE 손실을 건다.
추론 때는 그 자리를 모델이 autoregressive 로 생성한 뒤 flow matching 을 돌린다.

```
Task: {지시문}, State: {이산화 state}; Subtask: {문장};\nAction:
                                       ^^^^^^^^^^^^^^^ 여기에만 CE
```

`patches/apply_subtask_patch.py` 가 tokenizer 와 transform 을 고친다 (멱등, `--check` 지원).
GT 문장 15개는 `subtask/sentences.py` — 구간 이름을 그대로 쓰지 않고 자연어로 썼다.
PaliGemma 의 언어 prior 를 쓰고, 나중에 10개 태스크를 합칠 때
"close the left gripper" 같은 구절이 태스크를 넘어 공유되게 하기 위해서다.

재사용할 것은 이미 openpi 안에 다 있다 — 새 파라미터는 없다
(embedding tying: `Embedder.decode` 가 입력 임베딩 테이블을 전치해 쓴다).

| 필요한 것 | 있는 곳 |
|---|---|
| prelogits -> logits | `models/pi0_fast.py:216–225` |
| 토큰 CE + `token_loss_mask` | `models/pi0_fast.py:210–232` |
| autoregressive decode (KV cache, `lax.while_loop`) | `models/pi0_fast.py:241–300` |
| prompt 슬롯 | `models/tokenizer.py:28` |

**token budget 확인됨**: 현재 prompt 최대 153 토큰, 가장 긴 subtask 문장을 넣어도 163.
`max_token_len=200` 을 안 늘려도 된다.

**scheduled sampling**: 학습 내내 GT 를 넣으면 추론 때 자기 예측이 들어오는 순간 새
covariate shift 가 생긴다. `RSC_SUBTASK_SS_START` 부터 확률을 선형으로 올려 GT 대신
모델 예측을 넣는다. **기본값 0(비활성)** — 먼저 teacher forcing 으로 baseline 과 비교한 뒤 켠다.

### 진행 상황 (2026-09-26 02:50, step 22000 / 82000)

| step | `loss/flow` | `loss/subtask_ce` | `loss/subtask_acc_first` |
|---|---|---|---|
| 0 | 0.2958 | 6.3383 | 0.0000 |
| 1000 | 0.0511 | 0.0647 | 0.8825 |
| 5000 | 0.0131 | 0.0226 | 0.9475 |
| 10000 | 0.0079 | 0.0190 | 0.9650 |
| 20000 | 0.0052 | 0.0163 | 0.9600 |

`subtask_acc_first` 는 슬롯 첫 토큰의 정확도다. teacher forcing 에서 뒤쪽 단어는
언어 모델이면 그냥 맞히므로(전체 정확도 99.5%) **첫 토큰만이 "장면을 보고 phase 를
아는가"를 잰다.**

flow 비교 기준선: baseline 체크포인트 step 10000 을 같은 스크립트로 재면 **0.00421**.
(wandb 값과 직접 비교하면 안 된다 — 그쪽은 학습 중 랜덤 time 추출 값이다.)

### 구현 중 잡은 함정 세 가지

1. **prefix 가 완전 양방향이다.** `embed_prefix` 의 `ar_mask` 가 전부 False 라 위치 p 가
   토큰 p+1 을 이미 본다. 다음 토큰 예측 CE 가 '복사'로 풀린다 — step 0 의 7.29 가
   800스텝 만에 0.046 으로 무너지는 것으로 드러났다. `make_attn_mask` 가 샘플별 마스크를
   받으므로 **슬롯 자리만 causal 블록**으로 만들었다. 부작용으로 `compute_loss` 의
   `concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)` 이 깨지니 같이 고쳐야 한다.
2. **패딩까지 CE 를 걸면 40% 가 낭비다.** 문장+EOS 가 5~13토큰, 슬롯 14 → 패딩 39.5%.
   EOS(id 1)로 끝내고 그 뒤를 손실에서 뺐다. `token_loss_mask` 는 슬롯 전체를 덮은 채로
   둔다 — attention 의 causal 블록 경계로도 쓰이므로 줄이면 구조가 깨진다.
3. **teacher forcing 전체 정확도는 의미가 없다.** 98.66% 가 나와도 CE 는 2.36 이었다.

전체 기록은 [`results/findings_step2.md`](results/findings_step2.md).

### 복구

```
bash bootstrap_subtask.sh
```
빈 /workspace 에서 환경 → 패치 3종 → config → HF 체크포인트 회수 → 학습·업로더까지.
전부 멱등이다. 패치는 앵커가 안 맞으면 즉시 멈춘다(업스트림이 바뀐 것이므로).

### 비교 설계

학습 config 를 **82k cosine 그대로 유지**한다. 40k cosine 으로 짧게 돌리면 20k 시점의
learning rate 가 baseline 과 달라 같은 step 끼리 비교가 안 된다.
baseline 체크포인트는 10k 간격으로 전부 있다.

rollout 이 비결정적이라 짝지은 A/B 가 성립하지 않는다. 100개씩 독립 평가하고
Wilson CI 로 본다. baseline 이 1/100 이라 개선이 있으면 8–10/100 쯤에서 구분된다.

## 별건 — loss masking

`patches/apply_loss_mask.py`. pi05 는 `action_dim=32` 고정인데 우리 로봇은 14축이라
뒤 18축이 0 패딩이고, 손실이 32축을 마스크 없이 평균낸다.

**학습 결과는 바뀌지 않는다.** 패딩 축을 빼는 것은 손실 전체를 32/14 = 2.29배로
균일 스케일링하는 것과 같고 AdamW 는 기울기의 균일 스케일에 불변이다.
효용은 **로그 loss 를 런 간 비교 가능하게** 만드는 것 하나다.

축별 손실 몫은 1/32 균등이 아니라 **타깃 분산에 비례**한다 (`schedule/loss_share.py`).
items_handover 에서 좌그리퍼는 14축 중 **두 번째로 큰 축**(24.2%)이었다.
"그리퍼가 작은 축이라 못 배운다"는 추론은 틀렸다.

## 환경

- pi0.5 / openpi, LoRA (`gemma_2b_lora` + `gemma_300m_lora`)
- 동결되는 것은 `llm` 경로의 base weight 뿐이다. **SigLIP 비전 타워는 전체 학습된다**
  (경로에 `llm` 이 없어 freeze filter 에 안 걸린다)
- `action_horizon=50`, 실행은 `pi0_step=10`
- delta mask `make_bool_mask(6,-1,6,-1)` — 양팔 6축씩 delta, 그리퍼 2축은 절대값
