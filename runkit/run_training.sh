#!/usr/bin/env bash
# pi0.5 LoRA · click_bell · 40k 코사인 · 재계산+보정된 norm stats
# 체크포인트는 /workspace 에, 영속성은 HuggingFace 업로드로. /root 는 XFS inode
# 할당이 수백 MB 단위로 막혀서 체크포인트를 받지 못한다.
set -uo pipefail

PI05=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
# 런마다 다른 값은 환경변수로 받는다. 기본값은 click_bell 런이다.
CONFIG=${CONFIG:-pi05_robosyn_click_bell_lora_cos40k}
EXP=${EXP:-click_bell_cos40k}
TARGET_STEP=${TARGET_STEP:-39999}
CKPT_BASE=${CKPT_BASE:-/workspace/rsc_ckpt}
LOG=${LOG:-$CKPT_BASE/train_$EXP.log}
MAX_RESTARTS=20
MIN_GB=60
MIN_INODES=500000

mkdir -p "$CKPT_BASE"
cd "$PI05" || exit 1
set +u; source /root/.bashrc >/dev/null 2>&1; set -u

export HF_LEROBOT_HOME="$PI05/training_data"
export OPENPI_DATA_HOME=/workspace/.cache/openpi
export HF_HOME=/workspace/hf-home
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
# .bashrc 를 source 한 뒤라 거기 값이 들어와 있다. 호출자가 지정했으면 그걸 우선한다.
# 진행 중인 run 을 --resume 으로 이어붙일 때는 그 run 이 속한 엔티티를 그대로 써야 한다.
[ -n "${WANDB_ENTITY_OVERRIDE:-}" ] && export WANDB_ENTITY="$WANDB_ENTITY_OVERRIDE"
echo "[wandb] entity=$WANDB_ENTITY"
export WANDB_DIR=/workspace/wandb
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR"

step_dir="$CKPT_BASE/$CONFIG/$EXP"

preflight() {
  local gb inodes
  gb=$(df -BG --output=avail "$CKPT_BASE" | tail -1 | tr -dc '0-9')
  inodes=$(df --output=iavail "$CKPT_BASE" | tail -1 | tr -dc '0-9')
  echo "[preflight] $CKPT_BASE 여유 ${gb}GB · inode ${inodes}" | tee -a "$LOG"
  if [ "${gb:-0}" -lt "$MIN_GB" ] || [ "${inodes:-0}" -lt "$MIN_INODES" ]; then
    echo "[preflight] 부족 (요구 ${MIN_GB}GB / ${MIN_INODES} inode). 중단." | tee -a "$LOG"
    return 1
  fi
}

quick_fails=0
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  preflight || exit 1

  # 체크포인트가 하나라도 있으면 이어서 한다 (bootstrap 이 HF 에서 받아 둔다)
  if ls -1 "$step_dir" 2>/dev/null | grep -qE '^[0-9]+$'; then
    mode=(--resume); label="재개 (--resume)"
  elif [ "$attempt" -eq 0 ]; then
    mode=(--overwrite); label="첫 실행 (--overwrite)"
  else
    mode=(--resume); label="재시작 #$attempt (--resume)"
  fi
  echo "=== [$(date -Is)] $label ===" | tee -a "$LOG"

  start=$(date +%s)
  # 배치는 머신마다 다르다 (H100 32 / 이 3090 은 2~4). tyro 가 config 필드를
  # 그대로 덮어쓴다. 지정 안 하면 config 기본값을 쓴다.
  bs=(); [ -n "${BATCH:-}" ] && bs=(--batch-size "$BATCH")
  ./.venv/bin/python scripts/train.py "$CONFIG" \
      --exp-name="$EXP" \
      --checkpoint-base-dir="$CKPT_BASE" \
      "${bs[@]+"${bs[@]}"}" \
      "${mode[@]}" >> "$LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  echo "=== [$(date -Is)] train.py 종료 rc=$rc (${elapsed}초) ===" | tee -a "$LOG"
  [ "$rc" -eq 0 ] && { echo "학습 정상 완료" | tee -a "$LOG"; exit 0; }

  last=$(ls -1 "$step_dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "마지막 체크포인트: ${last:-없음}" | tee -a "$LOG"
  if [ -n "${last:-}" ] && [ "$last" -ge "$TARGET_STEP" ]; then
    echo "목표 스텝 도달." | tee -a "$LOG"; exit 0
  fi

  # 즉시 죽는 것은 환경/설정 문제다. 재시작 20번을 태워봐야 의미가 없다.
  if [ "$elapsed" -lt 120 ]; then
    quick_fails=$((quick_fails+1))
    echo "즉시 실패 ${quick_fails}회째" | tee -a "$LOG"
    [ "$quick_fails" -ge 3 ] && { echo "환경 문제로 보고 중단한다." | tee -a "$LOG"; exit 1; }
  else
    quick_fails=0
  fi
  sleep $(( 60 * (quick_fails + 1) ))
done
echo "재시작 한도 초과." | tee -a "$LOG"; exit 1
