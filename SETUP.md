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

# 2) 기존 체크포인트 받기 (권장 — norm_stats 가 같이 온다)
export WS=$HOME/rsc_ws                # 작업 루트. 원하는 데로.
export HF_TOKEN=hf_...
python3 $REC/fetch_subtask_ckpt.py \
  yai-robosync/pi05-items-handover-subtask-cos82k  $WS/ckpt

# 3) 1단계 — 성공 데모 + subtask + MEM 만. 실패 데이터는 아직 안 쓴다.
NORM_FROM_CKPT=$WS/ckpt/81999 bash $REC/setup/setup_train.sh --subtask --mem   # 약 20분

export PI05=$WS/RoboSynChallenge/policy/pi05
python3 $REC/mistake/prepare_posttrain_config.py $PI05 \
  pi05_robosyn_items_handover_subtask_cos82k  $WS/ckpt/81999  12000

CONFIG=pi05_robosyn_items_handover_subtask_cos82k_post12k \
PI05_SUBTASK_W=1.0 PI05_MEM_FRAMES=6 bash $REC/setup/train.sh
```

**1단계를 먼저 통과시킬 것.** 실패 데이터(mistake)는 정렬 복구와 라벨 파이프라인을
확인한 뒤 2단계에서 붙인다 — §4.2.

> `--recording` 은 **평가 중 롤아웃을 저장할 때만** 필요하다. 학습 설치에는 넣지 않는다.
> 녹화 패치(`apply_rollout_record2`)가 없는 깨끗한 upstream 에 정렬 패치만 걸면
> 앵커가 없어 설치가 거기서 멈춘다.

`setup_train.sh` 는 멱등하다. 중간에 끊겨도 다시 돌리면 된다. 마지막에
패치 검증(문자열)과 **config 로드 검증**(실제 import + norm_stats 확인)을 둘 다 돌린다.

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

### 3.4 norm_stats

**세 가지 방법이 있고, 첫 번째를 권한다.**

1. **기존 체크포인트에서 가져오기 (권장, 즉시)**
   체크포인트에는 그것이 학습에 쓴 통계가 `assets/` 에 동봉돼 있다. 이어서
   학습할 거라면 **그 통계를 그대로 써야** 입력 분포가 비트 단위로 보존된다.
   ```bash
   NORM_FROM_CKPT=$WS/ckpt/81999 bash $REC/setup/setup_train.sh --all
   ```

2. **새로 계산 (느리다)**
   ```bash
   cd $PI05 && HF_LEROBOT_HOME=$PI05/training_data \
     .venv/bin/python scripts/compute_norm_stats.py \
     --config-name pi05_robosyn_items_handover_lora_cos82k
   ```
   영상까지 디코딩해서 **몇 시간** 걸린다. `setup_train.sh` 는
   `CONFIRM_SLOW_NORM=1` 없이는 이 경로로 가지 않는다.

> 이전 판 문서에 있던 `--fast-fix` 플래그는 **upstream 에 존재하지 않는다.**
> 라이브 머신에만 있던 수정이 버전 관리에 안 들어간 것으로, 새 머신에서는
> 그대로 실패한다. 플래그를 지웠다. (리뷰 R5)

`setup_train.sh` 는 **선택한 config 전부**의 assets 경로에 통계를 놓는다.
예전에는 baseline 에만 만들어서 subtask/mistake config 로 데이터로더를 만들면
통계를 못 찾았다 (리뷰 R6). mistake config 는 `AssetsConfig(asset_id=...)` 로
데모 데이터셋 이름을 고정해, 병합(`_mix`) 데이터셋을 써도 같은 통계를 읽는다.

### 3.5 소스 패치 — **순서가 중요하다**

```bash
bash $REC/setup/apply_patches.sh $WS/RoboSynChallenge --all
```

**순서를 직접 쓰지 말고 이 스크립트를 쓸 것.** 패치끼리 앵커 의존이 있고,
순서를 틀리면 **앞쪽 파일만 바뀐 부분 적용 상태**로 멈춘다. 실제 의존 관계:

```
prepare_repo         cos40k TrainConfig    -> prepare_handover 의 삽입 기준점
prepare_handover     state_key/image_key_* -> MEM wire2 의 앵커
apply_loss_mask      (독립)
subtask 3종          tokenizer/transforms/policy/pi0/gemma/config
prepare_subtask      repack 에 frame_index -> apply_mistake 의 앵커
apply_mistake        repack 에 episode_index + 프롬프트 Mistake 필드
MEM                  wire2 가 repack 에 *_is_pad
prepare_mistake      병합 데이터셋 TrainConfig (데이터셋이 있어야 한다)
```

예전 `setup_train.sh` 는 mistake 를 `prepare_subtask_config` **앞에** 돌려서
`frame_index` 앵커 0개로 멈췄다 (리뷰 R3). 전부 **멱등**이고 대부분 `--check` 를 지원한다.

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

먼저 데이터셋을 만들어야 한다 (실패 롤아웃이 필요). **순서가 중요하다** —
HF 에 올려 둔 롤아웃은 **정렬 복구 전(raw)** 이다.

```bash
export PI05=$WS/RoboSynChallenge/policy/pi05

# ① raw 롤아웃 100개를 받는다 (직접 수집하려면 rollout/collect_subtask.sh)
huggingface-cli download yai-robosync/handover-rollouts --repo-type dataset \
  --include "subtask/*" --local-dir $WS/rollouts

# ② (obs, action) 정렬 복구 — 이걸 건너뛰면 한 스텝 어긋난 데이터로 학습한다
$PI05/.venv/bin/python $REC/rollout/fix_rollout_alignment.py \
  $WS/rollouts/subtask  $WS/rollouts/subtask_aligned
#    이미 복구된 데이터를 또 넣으면 meta 의 rsc_alignment_fixed 를 보고 멈춘다

# ③ 라벨 + 절단 지점 (복구본 기준으로 다시 계산해야 한다)
$PI05/.venv/bin/python $REC/mistake/mistake_labels.py $WS/rollouts/subtask_aligned

# ④ 데모 + 롤아웃 병합
PI05=$PI05 ROLLOUTS=$WS/rollouts/subtask_aligned \
  $PI05/.venv/bin/python $REC/mistake/merge_mistake_ds.py

# ⑤ TrainConfig 2개 등록 + 통계 배치
python3 $REC/mistake/prepare_mistake_config.py $PI05
for C in pi05_robosyn_items_handover_mistake_cos82k \
         pi05_robosyn_items_handover_subtask_mistake_cos82k; do
  D=$PI05/assets/$C/RoboSynChallenge/cobotmagic_Sim_items_handover
  mkdir -p $D && cp $WS/ckpt/81999/assets/RoboSynChallenge/cobotmagic_Sim_items_handover/norm_stats.json $D/
done

# ⑥ 검증
cd $PI05 && env PYTHONPATH=src HF_LEROBOT_HOME=$PI05/training_data PI05_MISTAKE=1 PI05_SUBTASK_W=1.0 \
  .venv/bin/python $REC/patches/verify_configs.py \
  pi05_robosyn_items_handover_mistake_cos82k pi05_robosyn_items_handover_subtask_mistake_cos82k
```

`setup_train.sh --all` 은 병합 데이터셋이 없으면 mistake TrainConfig 를 만들지
않고 검증 목록에서도 뺀다. 위 ①~⑥ 을 끝낸 뒤 `apply_patches.sh` 를 다시 돌리거나
⑤ 를 직접 실행하면 등록된다.

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
HF 에 있으니 거기서 10~15k 만 붙이는 쪽이 훨씬 싸다. MEM 은 파라미터를 0개
추가하고 `T=1` 에서 항등이라 **기존 체크포인트가 유효한 출발점**이고,
mistake 도 드롭아웃 5% 경로가 기존 프롬프트와 바이트 단위로 같다.

**`--resume` 으로는 안 된다.** 두 가지 이유가 있다 (리뷰 R11):

- 모든 config 의 `weight_loader` 가 `pi05_base` 를 가리킨다. 새 experiment 로
  띄우면 81999 가 아니라 **base 에서 시작한다.** 체크포인트 파일을 받아 두는
  것만으로는 아무 효과가 없다.
- 완료된 82k 체크포인트를 같은 config 로 `--resume` 하면 `train_state.step` 이
  이미 82000 이라 `range(start_step, num_train_steps)` 가 비고 **한 스텝도 안 돈다.**
  (저장 전에 step 이 1 증가해서 디렉터리 이름 81999 와 state.step 82000 이 어긋난다.)

그래서 **별도 posttrain config** 를 만든다. 가중치만 가져오고 optimizer·step 은
새로 초기화하며, 짧은 warmup/decay 를 새로 건다. 통계는 그 체크포인트가 쓰던
것을 `AssetsConfig(assets_dir=<ckpt>/assets)` 로 그대로 가리킨다.

```bash
python3 $REC/fetch_subtask_ckpt.py \
  yai-robosync/pi05-items-handover-subtask-cos82k  $WS/ckpt

python3 $REC/mistake/prepare_posttrain_config.py $PI05 \
  pi05_robosyn_items_handover_subtask_mistake_cos82k  $WS/ckpt/81999  12000
# -> pi05_robosyn_items_handover_subtask_mistake_cos82k_post12k
#    weight_loader = <ckpt>/81999/params, 12k steps, warmup 600, 1e-5 -> 1e-6
```

`--resume` 은 **같은 실험의 장애 복구에만** 쓴다. 새 데이터·새 설정은 언제나
weights-only warm start 다.

사용 가능한 체크포인트:

| HF repo | 설명 |
|---|---|
| `yai-robosync/pi05-items-handover-cos82k` | baseline 82k |
| `yai-robosync/pi05-items-handover-subtask-cos82k` | subtask 82k |
| `yai-robosync/handover-rollouts` | 실패 롤아웃 100개 (mistake 라벨용) |

---

## 6. GPU 별 배치 가이드

> **이 숫자로 production batch 상한을 정하지 말 것.** (리뷰 R12)
> 아래는 `tests/mem_speed.py` 벤치마크 하니스의 값이다. 실제 `train.py` 와
> **다른 점이 있다**: 체크포인트를 전부 bf16 으로 복원하고(실제는 모델 초기화의
> param dtype 에 맞춰 병합하고 frozen 만 bf16), optimizer 도 config 의 것이 아니라
> `optax.adamw(1e-5)` 를 쓴다(clipping·b2 가 다르다). 경향 파악용으로만 보고,
> 상한은 §7 처럼 **실제 train.py** 로 재는 것이 맞다.

RTX 3090 24GB, batch 는 global batch (gradient accumulation 구현 없음):

| T | batch | 스텝 시간 | 활성화 메모리 | 82k 환산 | |
|---|---|---|---|---|---|
| 1 | 4 | 1,442 ms | 10.77 GiB | 32.8 h | OK |
| 1 | 8 | 2,623 ms | 11.52 GiB | 59.7 h | OK |
| 1 | 16 | — | 14.81 GiB | — | OOM |
| 6 | 1 | — | 10.88 GiB | — | OOM (단편화) |
| 6 | 2 | 1,314 ms | 11.24 GiB | 29.9 h | OK |
| 6 | 4 | — | 11.88 GiB | — | OOM |

활성화 메모리에 파라미터(bf16 3.4B ≈ 6.8 GiB)와 AdamW 모멘트(trainable 467M ×
2 × 4B ≈ 3.7 GiB)를 더한 값이 실제 점유량이다. 24 GiB 에서 11~12 GiB 활성화는
경계선이고, `T=6 batch=1` 이 OOM 나고 `batch=2` 가 통과한 것이 단편화의 증거다.
**80 GB 로 가면 이 경계가 통째로 사라진다** — 거기서는 §7 로 2→4→8→16 을 직접
재고 정하는 것이 맞다.

**배치를 키울 때 러닝레이트.** 위 config 들은 batch 4 기준으로 코사인 스케줄이
짜여 있다. 먼저 **기존 LR 그대로** 짧게 돌려 비교하는 편이 해석이 쉽다 — √N 배를
자동으로 올리면 배치 효과와 LR 효과가 섞인다. 그리고 **다른 batch 에서 같은
step 의 loss 를 비교하는 것은 같은 데이터 노출량을 비교한 것이 아니다.**
처리한 sample 수와 update 수를 함께 적어 둘 것.

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
| `patches/verify_patches.py --all` | 패치가 전부 들어갔나 (문자열 검사) |
| `patches/verify_configs.py <config>...` | config 가 실제로 **로드되고** norm_stats 를 읽나 (**필수**) |
| `tests/decode_eos_test.py` | EOS 뒤가 pad 로 채워지나 |
| `tests/mem_infer_path_test.py` | 추론 이미지 shape · frame_valid 통과 |
| `tests/mem_buffer_evalorder_test.py` | **실제 평가 호출 순서**에서 버퍼 간격이 맞나 |
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

**그리고 문자열 검사는 기능 검사가 아니다.** `verify_patches.py --all` 이 OK 를
내는데도 `config.py` 가 import 조차 안 되는 상태였던 적이 있다 (`DataConfig` 에
없는 필드를 넣었다). 그래서 `verify_configs.py` 를 따로 둔다 — 실제로 import 하고
`DataConfig` 를 만들어 `norm_stats` 가 **읽혔는지**까지 본다.

**아직 자동화되지 않은 것:** 실제 `train.py` 2-step smoke, 100-step steady-state
GPU 측정, save→resume→추론 로딩 왕복. 유료 장시간 학습 전에 이 셋을 손으로라도
돌릴 것.

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
    테스트도 마찬가지다 — FAIL 을 찍고 `exit 0` 으로 끝나면 자동 게이트가 안 된다.

12. **`set -euo pipefail` 아래에서 없는 디렉터리에 `find` 를 걸지 말 것.**
    `N=$(find /없는/경로 -name '*.x' | wc -l)` 는 `wc` 가 0 을 찍어도 `find` 의
    실패가 `pipefail` 에 걸려 **셸이 그 자리에서 죽는다.** stderr 까지 버리면
    이유도 안 보인다. 디렉터리 존재를 먼저 확인할 것 (`count_files()` 참고).

13. **f-string 패치 템플릿에서 중괄호는 두 배로.** `}}` 는 `}` 하나가 된다.
    dict 두 개를 닫는 자리를 한 개로 줄여 괄호가 안 맞는 파일을 만든 적이 있다.

14. **녹화기는 `step()` 이 돌려준 obs 를 쓰면 안 된다.** 그건 실행 **후** 관측이라
    `(o_{t+1}, a_t)` 가 저장된다. BC 는 `(o_t, a_t)` 가 필요하다. step 직전 관측을
    따로 들고 있다가 짝지을 것. 환경이 버퍼를 재사용할 수 있으니 복사해서 보관한다.
    **검증 방법:** `(action-state)` 와 `(다음 state-state)` 의 코사인을 본다.
    올바르면 데모와 비슷한 +0.9 대가 나온다. `|action-state|` 크기만 비교하면
    컨트롤러 추종 지연에 가려져 판별이 안 된다.

---

## 10. 외부 코드 리뷰 반영 (2026-10-04)

Runpod 이전 전에 받은 리뷰에서 지적된 것들. **확인한 항목은 전부 재현됐다.**

| | 문제 | 상태 |
|---|---|---|
| R1 | 녹화가 `(o_{t+1}, a_t)` 를 저장 — 한 스텝 어긋남 | 고침 + 기존 데이터 복구 |
| R2 | subtask EOS 뒤 임의 토큰이 action 조건에 들어감 | 고침 + 테스트 |
| R3 | 자동 setup 의 패치 순서가 의존성과 반대 | `apply_patches.sh` 로 통합 |
| R4 | 없는 디렉터리에 `find` → `pipefail` 로 셸 종료 | `count_files()` |
| R5 | `--fast-fix` 가 upstream 에 없음 | 제거, 체크포인트 통계 재사용 |
| R6 | subtask/mistake config 의 norm_stats 미준비 | `AssetsConfig` + 전 config 배치 |
| R7 | MEM 패치 8개가 경로를 무시, infer 는 루트 계약이 다름 | PI05 루트로 통일 |
| R8 | 추론 4D transpose · `frame_valid` 유실 | 고침 + 테스트 |
| R9 | 버퍼 시간이 env step 이 아니라 호출 횟수 | `mem_tick()` + 중복 제거 |
| R10 | MEM 에서 train.py 첫 배치 이미지 로깅 실패 | 현재 프레임만 로깅 |
| R11 | 81999 에서 이어 학습하는 경로 없음 | `prepare_posttrain_config.py` |
| R12 | 벤치마크 dtype·optimizer 가 실제와 다름 | §6 에 경고 명시 |
| R13 | 녹화 재시작이 사이드카를 덮어씀 | 기존 meta 읽고 atomic write |
| R14 | 데이터셋 재생성해도 config 라벨이 안 바뀜 | marker 블록 교체 |

### R1 복구 — 어떻게 확인했나

처음에 `|action - state|` 크기로 판별하려 했는데 **컨트롤러 추종 지연에
가려져 결론이 안 났다** (수정 후 오히려 커졌다). 방향으로 보면 명확하다:

```
(action - state) 와 (다음 state - state) 의 코사인 중앙값
  대본 데모         +0.918      <- action 이 그 행의 state 에서 앞으로 가는 명령
  롤아웃 (수정 전)   +0.370      <- 규약 불일치
  롤아웃 (수정 후)   +0.800      <- 데모와 같아짐
```

복구는 `action` 열만 한 칸 당기고 마지막 행을 버린다
(`rollout/fix_rollout_alignment.py`). 잃는 것은 `(o_0, a_0)` 한 쌍인데,
`o_0`(리셋 직후)은 애초에 기록되지 않았다. 영상은 그대로 두고 parquet 만 줄인다.

복구 후 `mistake_labels` 를 다시 돌리면 `t_fail` 중앙값이 86 → 85 로 한 프레임
움직인다. 분류(A 92 / B 5 / C 2 / D 1)는 그대로다.

### 재리뷰 (72d7e96) 에서 추가로 나온 것

| | 문제 | 상태 |
|---|---|---|
| N1 | posttrain 생성기가 marker 를 같이 복사해 config.py 를 SyntaxError 로 만들고 **exit 0** | AST 로 호출 범위 추출 · 쓰기 전 `ast.parse` · atomic write |
| N2 | fresh upstream 에 녹화 패치 없이 정렬 패치를 걸어 설치가 멈춤 | `--recording` 옵션, record2 → align 순서 |
| N3 | `--all` 이 병합 데이터셋 없이 mistake config 를 검증 | 없으면 목록에서 제외 + §4.2 2단계 절차 |
| N4 | `"${OPTS[@]:-}"` 가 빈 배열에서 **빈 문자열 1개**를 만듦 | `"${OPTS[@]}"` |
| N5 | norm_stats 를 자기 자신에게 `cp` → exit 1 → `set -e` 로 중단 | `-ef` 로 같은 파일이면 건너뜀 |

fresh worktree(upstream 9815e9e)에서 `--all`, `--all --recording`, 재실행,
mix 유/무 네 경우를 실제로 돌려 확인했다.

### 아직 남은 설계 과제

- **`_mix` 의 subtask 라벨은 성공 데모의 고정 스케줄을 쓴다.** 실패한 grasp
  뒤에도 시간상 `lift the pen` 같은 지시문이 붙는다. "시도 중인 동작"을
  라벨링할지 "완료 상태/회복 단계"를 라벨링할지 먼저 정해야 한다.
- **`mistake=true` 도 같은 방향의 positive flow-matching loss 를 받는다.**
  이 플래그가 실패 action 을 배척하지도, 회복 action 을 가르치지도 않는다.
  추론을 `false` 로 조건화하는 분리 학습 실험일 뿐이다. 복구 데이터가 없는
  상태에서 회복 행동을 기대할 근거는 없다.
- **첫 subtask 토큰 정확도는 phase 정확도가 아니다.** `move…` 처럼 여러 phase 가
  첫 토큰을 공유한다. 전체 문장을 phase ID 로 매핑해 재야 한다.
- **FAST action CE, Knowledge Insulation, bbox 의 VLA 입력 연결은 아직 없다.**
  `detect/` 는 라벨 생성·탐지 평가 도구이고, box 를 학습 프롬프트나 모델에
  넣는 경로가 아니다.
