# 다른 머신에서 학습 돌리기

VESSL 파드(RTX 3090 24GB) 밖, 더 큰 GPU 에서 이 실험을 재현하기 위한 안내.
**처음부터 끝까지 복붙으로 돌아가게** 쓰려고 했다. 막히면 §9 함정 모음을 먼저 볼 것.

- 베이스 모델: pi0.5 (openpi), LoRA
- 과제: RoboSynChallenge `items_handover`
- 데이터: 대본 시연 1,000 에피소드 (327,000 프레임, 25fps, 에피소드당 327 프레임 고정)
- 올릴 수 있는 것 세 가지: **subtask 생성 / mistake 메타데이터 / MEM 시간축 메모리**

---

## 1. 요구 사항

| | 최소 | 권장 |
|---|---|---|
| GPU | 24 GB (T=1 batch 4 만) | 80 GB (A100/H100) — MEM T=6 을 큰 배치로 |
| 디스크 | 60 GB | 120 GB (체크포인트 9.6GB × 여러 개) |
| CUDA 드라이버 | 12.1+ | 12.4+ |
| Python | 3.11 (uv 가 설치) | |
| 시스템 | `ffmpeg` `build-essential` `pkg-config` | |

시뮬레이터(EmbodiChain)는 **학습에 필요 없다.** 평가할 때만 필요하고, 그건
`dexforce/embodichain:ubuntu22.04-cuda12.8` 컨테이너가 따로 필요하다.
여기 안내는 **학습만** 다룬다.

디스크 내역: 데이터 8.5GB + venv 약 12GB + pi0.5 base 가중치 7GB + 체크포인트 9.6GB/개.

---

## 2. 빠른 시작

```bash
# 0) 이 저장소
git clone https://github.com/YAI-VLA1/pi0.5-robosynchallenge-subtask.git ~/rsc
export REC=~/rsc

# 1) 시스템 패키지
sudo apt-get update && sudo apt-get install -y ffmpeg build-essential pkg-config
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv

# 2) 전체 세팅 (저장소 + venv + 데이터 + norm_stats + 패치 전부)
export WS=$HOME/rsc_ws                # 작업 루트. 원하는 데로.
export HF_TOKEN=hf_...                # 데이터/체크포인트 접근용
bash $REC/setup/setup_train.sh        # 약 25분 (데이터 12분 + venv 6분 + norm_stats 2분)

# 3) 패치가 전부 들어갔는지 하드 체크 — 반드시 통과시킬 것
python3 $REC/patches/verify_patches.py $WS/RoboSynChallenge --all

# 4) 학습
bash $REC/setup/train.sh
```

`setup_train.sh` 는 멱등하다. 중간에 끊겨도 다시 돌리면 된다.

---

## 3. 단계별 (수동으로 할 때)

### 3.1 베이스 저장소

```bash
mkdir -p $WS && cd $WS
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
cd RoboSynChallenge && git checkout 9815e9e
git apply $REC/setup/robosync_fixes.patch
```

커밋을 고정하는 이유: 업스트림이 바뀌면 소스 패치 앵커가 깨진다.
패치 스크립트는 앵커가 안 맞으면 **조용히 넘어가지 않고 멈춘다**.

### 3.2 venv

```bash
python3 $REC/setup/prepare_repo.py $WS/RoboSynChallenge/policy/pi05
cd $WS/RoboSynChallenge/policy/pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync                          # 약 6분
cp -r src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
uv pip install --python .venv/bin/python hf_xet        # HF 전송 가속 (체크포인트 9.6GB)
```

`prepare_repo.py` 가 `pyproject.toml` 에서 `dexsim-engine` / `embodichain` /
`robosynchallenge` 를 뺀다. `pymeshlab` 이 glibc 2.35 휠만 내서 구형 이미지에서
`uv sync` 가 통째로 막히기 때문이고, 셋 다 학습에 안 쓴다.

확인:

```bash
.venv/bin/python -c "import openpi.training.data_loader, jax; print(jax.devices())"
```

### 3.3 데이터 (8.5 GB)

```bash
export PI05=$WS/RoboSynChallenge/policy/pi05
export DATA=$PI05/training_data/RoboSynChallenge/cobotmagic_Sim_items_handover
$PI05/.venv/bin/python $REC/setup/fetch_data.py        # 재시도 루프 포함
```

**완료 판정을 `meta/info.json` 존재로 하면 안 된다.** 그 파일이 맨 먼저 도착해서
2.5GB(영상 580/3000)에서 완료로 착각한다. parquet 1000 + mp4 3000 을 셀 것:

```bash
find $DATA/data -name '*.parquet' | wc -l     # 1000
find $DATA/videos -name '*.mp4'   | wc -l     # 3000
```

### 3.4 학습 설정 + norm_stats

```bash
python3 $REC/setup/prepare_handover.py $PI05        # TrainConfig 2개 삽입
cd $PI05 && HF_LEROBOT_HOME=$PI05/training_data \
  .venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robosyn_items_handover_lora_cos82k --fast-fix
```

`--fast-fix` 는 parquet 에서 state/action 만 읽는다 (2분). 일반 경로는 영상까지
디코딩해서 **3시간 18분** 걸린다.

### 3.5 소스 패치 — **순서가 중요하다**

```bash
R=$WS/RoboSynChallenge
python3 $REC/patches/apply_loss_mask.py      $R   # ① 항상
python3 $REC/subtask/apply_subtask_patch.py  $R   # ② subtask 쓸 때
python3 $REC/subtask/apply_decode_patch.py   $R   # ③
python3 $REC/subtask/apply_subtask_v2.py     $R   # ④
python3 $REC/subtask/prepare_subtask_config.py $PI05 $REC/schedule/schedule_sim.json
python3 $REC/mistake/apply_mistake.py        $PI05 # ⑤ mistake 쓸 때
bash    $REC/mem/apply_all.sh                $PI05 # ⑥ MEM 쓸 때
```

②③④ 는 이 순서로만 검증했다. ①⑤⑥ 는 서로 독립이고 순서 무관이다.
전부 **멱등**이고 `--check` 를 지원한다 (⑥ 제외).

| 패치 | 하는 일 | 없으면 생기는 일 |
|---|---|---|
| `apply_loss_mask` | action 32축 중 **뒤 18축(0 패딩)을 손실에서 제외** | 손실의 56%가 "0을 맞혀라". 증상 없이 학습 효과만 희석 |
| `apply_subtask_patch` | subtask 슬롯 토크나이즈 · CE 손실 · attention 마스크 · dict 통과 | subtask 문장이 모델에 안 들어감 |
| `apply_decode_patch` | 추론 때 슬롯을 모델이 채우게 | 평가에서 subtask 생성 불가 |
| `apply_subtask_v2` | Task 지시문 제거 · EOS · 분리 로깅 | 프롬프트 형식이 체크포인트와 불일치 |
| `apply_mistake` | 프롬프트에 `Mistake: true/false` | mistake 조건화 없음 |
| MEM 패치 | SigLIP 에 space-time 분리 어텐션 | T>1 이 무시됨 |

### 3.6 패치 검증 — 건너뛰지 말 것

```bash
python3 $REC/patches/verify_patches.py $R --all
```

**이게 왜 있냐면:** 2026-10-04 에 bootstrap 이 `if [ ! -d "$R" ]` 안에서만
`apply_loss_mask.py` 를 불러서, 저장소가 이미 있는 경로로 재구축했더니 JAX 쪽
손실 마스킹이 통째로 빠졌다. **증상이 없다** — 학습은 그냥 돌고 손실만 희석된다.
같은 날 subtask CE 와 subtask 통과도 빠져 있었다.

---

## 4. 실험 옵션 세 가지

세 개는 서로 독립이다. 환경변수로 켜고 끈다.

### 4.1 subtask 생성

프롬프트: `State: <14개 정수>; Subtask: <14토큰 슬롯>;\nAction: `

모델이 슬롯을 스스로 채우고(CE 손실), 그 조건 위에서 action 을 낸다.
라벨은 **사람이 안 달았다** — sim 에피소드가 `action_config.json` 의 키포즈 DAG
재생이라 스케줄을 풀면 구간 경계가 스텝 단위로 나온다 (`schedule/schedule.py`).

```bash
export PI05_SUBTASK_W=1.0      # CE 가중치. 0 이면 완전히 꺼진다
```

### 4.2 mistake 메타데이터 (pi0.7 방식)

프롬프트에 `Mistake: true` / `Mistake: false` 를 넣는다.
실패 롤아웃에서 **실수 시점 50프레임 전부터** true 다.

먼저 데이터셋을 만들어야 한다 (실패 롤아웃이 필요):

```bash
# 롤아웃 100개를 HF 에서 받는다 (직접 수집하려면 rollout/collect_subtask.sh)
huggingface-cli download yai-robosync/handover-rollouts --repo-type dataset \
  --include "subtask/*" --local-dir /tmp/rollouts

python3 $REC/mistake/mistake_labels.py      /tmp/rollouts/subtask    # 라벨 + 절단 지점
python3 $REC/mistake/merge_mistake_ds.py                             # 데모+롤아웃 병합
python3 $REC/mistake/prepare_mistake_config.py $PI05                 # TrainConfig 2개
```

`merge_mistake_ds.py` 는 데모 parquet·영상을 **심볼릭 링크**로 둔다 (복사 없음).
롤아웃 parquet 98개만 새로 쓴다. 결과: 1,098 에피소드 / 341,252 프레임,
mistake=true 9,797 프레임 = **2.9%**.

```bash
export PI05_MISTAKE=1          # 학습·추론 양쪽에 같은 값을 줄 것
```

**드롭아웃 5%** 가 들어간다 (pi0.7 과 같다). 드롭되면 필드가 통째로 빠지는데,
그 형태가 **기존 프롬프트와 바이트 단위로 같다** — 기존 체크포인트에서
이어붙일 때 5%는 모델이 알던 입력 그대로다.

검증:

```bash
cd $PI05 && env PYTHONPATH=src HF_LEROBOT_HOME=$PI05/training_data \
  PI05_MISTAKE=1 .venv/bin/python $REC/mistake/mistake_smoke.py
# 토큰을 문자열로 되돌려 6항목을 확인한다. MISTAKE-SMOKE OK 가 나와야 한다.
```

### 4.3 MEM (시간축 메모리)

SigLIP **4층마다** space-time 분리 어텐션을 넣는다. **새 파라미터가 0개**다.

```bash
export PI05_MEM_FRAMES=6         # 과거 프레임 수 (1 = 끄기, 비트 단위로 기존과 동일)
export PI05_MEM_STRIDE_S=1.0     # 프레임 간격 (초)
export PI05_MEM_HIST_DROPOUT=0.3 # 히스토리 통째 드롭 확률 (논문값)
```

자세한 것은 [`mem/README.md`](mem/README.md).

---

## 5. 학습 실행

### 5.1 환경변수 한눈에

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `PI05_REAL_ACTION_DIM` | `14` | 손실에 쓸 실제 축 수. `0` 이면 마스킹 끔 |
| `PI05_SUBTASK_W` | `0.0` | subtask CE 가중치. `>0` 이면 슬롯도 생긴다 |
| `PI05_MISTAKE` | `0` | `1` 이면 프롬프트에 Mistake 필드 |
| `PI05_MEM_FRAMES` | `1` | 과거 프레임 수 |
| `PI05_MEM_STRIDE_S` | `1.0` | 프레임 간격(초) |
| `PI05_MEM_HIST_DROPOUT` | `0.3` | 히스토리 드롭 확률 |
| `HF_LEROBOT_HOME` | — | `$PI05/training_data` (필수) |
| `OPENPI_DATA_HOME` | — | 캐시 경로 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | GPU 메모리 상한 비율 |

**학습과 추론에 같은 값을 줘야 한다.** 프롬프트 형식이 달라지면 조용히 안 죽고
loss 만 튄다 (2026-09-26 에 0.015 → 10.25 로 튀었다).

### 5.2 config 고르기

| config | 데이터셋 | subtask | mistake |
|---|---|---|---|
| `pi05_robosyn_items_handover_lora_cos82k` | 데모 1,000 | ✗ | ✗ |
| `pi05_robosyn_items_handover_subtask_cos82k` | 데모 1,000 | ✓ | ✗ |
| `pi05_robosyn_items_handover_mistake_cos82k` | 병합 1,098 | ✗ | ✓ |
| `pi05_robosyn_items_handover_subtask_mistake_cos82k` | 병합 1,098 | ✓ | ✓ |

전부 82,000 스텝 코사인 · batch 4 · LoRA(gemma_2b_lora + gemma_300m_lora) ·
`action_horizon=50` 이라 **같은 step 끼리 비교할 수 있다**.

82k 를 고른 이유: 327,000 프레임 / batch 4 = **1 에포크 = 81,750 스텝**이다.

### 5.3 실행

```bash
cd $PI05
export HF_LEROBOT_HOME=$PI05/training_data
export OPENPI_DATA_HOME=$WS/.cache/openpi
export PI05_REAL_ACTION_DIM=14
export PI05_SUBTASK_W=1.0 PI05_MISTAKE=1 PI05_MEM_FRAMES=6

.venv/bin/python scripts/train.py \
  pi05_robosyn_items_handover_subtask_mistake_cos82k \
  --exp-name my_run --overwrite
```

이어서 하려면 `--resume` (그리고 체크포인트 디렉터리에 `wandb_id.txt` 가 있어야 한다 —
없으면 train.py 가 죽는다).

### 5.4 처음부터 vs 기존 체크포인트에서 이어붙이기

82k 를 처음부터 돌리면 3090 기준 **33시간**이다. 이미 학습된 81999 체크포인트가
HF 에 있으니, 거기서 10~15k 스텝만 붙이는 쪽이 훨씬 싸다:

```bash
python3 $REC/fetch_subtask_ckpt.py \
  yai-robosync/pi05-items-handover-subtask-cos82k  $CKPT_DIR
```

MEM 은 파라미터를 0개 추가하고 `T=1` 에서 항등이라 **기존 체크포인트가 유효한
출발점**이다. mistake 도 드롭아웃 5% 경로가 기존 프롬프트와 같다.

사용 가능한 체크포인트:

| HF repo | 설명 |
|---|---|
| `yai-robosync/pi05-items-handover-cos82k` | baseline 82k |
| `yai-robosync/pi05-items-handover-subtask-cos82k` | subtask 82k |
| `yai-robosync/handover-rollouts` | 실패 롤아웃 100개 (mistake 라벨용) |

---

## 6. GPU 별 배치 가이드

RTX 3090 24GB 실측. 더 큰 GPU 는 외삽이다 — 돌려 보고 §7 로 직접 재는 쪽을 권한다.

<!-- SWEEP_TABLE -->

**배치를 키울 때 러닝레이트도 같이 봐야 한다.** 위 config 들은 batch 4 기준으로
코사인 스케줄이 짜여 있다. batch 를 N 배로 키우면 스텝 수를 N 분의 1로 줄이고
(`num_train_steps`, `decay_steps` 둘 다) peak lr 을 √N 배 정도로 올리는 것이
무난하다. `TrainConfig` 를 복사해 새 이름으로 등록할 것.

---

## 7. 직접 재보기

```bash
cd $PI05
# 스텝 시간 + 82k 환산 + 메모리
env PYTHONPATH=src HF_LEROBOT_HOME=$PI05/training_data \
  PI05_MEM_FRAMES=6 MEM_BATCH=8 MEM_STEPS=10 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python $REC/tests/mem_speed.py
```

출력의 `reduced to NN GiB` 가 스텝 하나가 쓰는 활성화 메모리다.
여기에 파라미터(bf16 3.4B ≈ 6.8 GiB)와 AdamW 모멘트(trainable 467M × 2 × 4B ≈
3.7 GiB)를 더한 값이 실제 점유량이다.

---

## 8. 검증 스크립트

학습을 띄우기 전에 돌릴 것. 전부 `/tests` 와 각 디렉터리에 있다.

| 스크립트 | 보는 것 |
|---|---|
| `patches/verify_patches.py --all` | 패치가 전부 들어갔나 (**필수**) |
| `tests/mem_equiv.py` | `T=1` 이 기존과 비트 단위로 같은가 |
| `tests/mem_t_tests.py` | causal · padding 무영향 · 과거→현재 전달 |
| `tests/mem_t_f64.py` | float32 누적오차인지 논리 오류인지 가름 |
| `tests/mem_data_test.py` | delta_timestamps 와 `_is_pad` 배선 |
| `tests/mem_train_step.py` | forward+backward 가 실제로 도는가 |
| `tests/mem_buffer_test.py` | 추론 프레임 버퍼 |
| `mistake/mistake_smoke.py` | 프롬프트 문자열 6항목 |
| `subtask/smoke.py` | subtask 주입·프롬프트 형식·EOS |

**shape 가 맞는 것과 학습이 도는 것은 다르다.** MEM 구현에서 예외 없이 조용히
망가지는 버그가 6개 나왔고(그중 하나는 `cam_high` 를 212×5 로 뭉갰다),
전부 `mem_train_step.py` 를 실제로 돌려서야 잡혔다.

---

## 9. 함정 모음

같은 데서 두 번 이상 당한 것들.

1. **패치 앵커 유일성을 먼저 검사할 것.** `config.py` 의 repack 블록과
   `libero_policy.py` 의 `if "prompt" in data:` 는 여러 클래스에 똑같이 나온다.
   첫 일치를 잡으면 엉뚱한 클래스에 들어가고, 증상이 "패치했는데 아무 일도 안 일어남"
   이라 찾기 어렵다. 클래스 범위로 좁힐 것.

2. **앵커는 좁게 잡을 것.** `return inputs` 처럼 한 줄이면 다른 패치가 그 위에
   줄을 끼워도 안 깨진다. 넓게 잡으면 패치끼리 서로 앵커를 깨서 적용 순서에
   종속된다.

3. **`s.replace()` 뒤에 `assert` 를 걸 것.** 없으면 조용히 실패한다.

4. **import 순서 segfault.** `openpi.training.config` 를 먼저 올린 뒤
   `data_loader` 를 올리면 죽는다 (exit 139, 로그 없음). `scripts/train.py` 의
   preamble 순서를 그대로 복사할 것 — 이 저장소의 테스트 스크립트들이 전부 그렇게 돼 있다.

5. **FFmpeg 없으면 `torchcodec` 이 `libavutil.so.56` 을 못 찾는다.**

6. **영상이 AV1 이라 OpenCV 가 못 읽는다.** ffmpeg rawvideo 파이프를 쓸 것
   (`detect/yolo_label_dataset.py` 가 예시).

7. **`np.savez` 는 파일명이 `.npz` 로 안 끝나면 확장자를 덧붙인다.** 임시 파일을
   `x.npz.tmp` 로 두면 `x.npz.tmp.npz` 가 생겨서 rename 이 실패한다.

8. **LeRobot 은 `chunk = episode_index // chunks_size` 로 경로를 만든다.**
   `chunks_size=1000` 이면 에피소드 1000번부터는 `chunk-001` 이다.

9. **병합 데이터셋의 `episodes_stats.jsonl` 은 feature 별 shape 가 전부 같아야 한다.**
   아니면 `aggregate_stats` 가 `np.stack` 에서 터진다. 학습이 안 읽는 키는 빼면 된다.

10. **로컬 전용 데이터셋은 Hub 조회에서 막힌다.** `lerobot 0.1.0` 의
    `LeRobotDataset` 이 생성자에서 `get_safe_version()` 으로 Hub 에 붙는다.
    `HF_HUB_OFFLINE=1` 은 해결이 아니다 (`OfflineModeIsEnabled` 로 바뀔 뿐).
    `setup/lerobot_offline.py` 의 `patch()` 를 쓸 것.

11. **복구 스크립트에 "조용히 넘어가는 경로"를 하나도 두지 말 것.** 복구는 평소에
    실행되지 않아 검증이 안 된다. 실패하면 반드시 종료 코드 1 로 멈출 것.
