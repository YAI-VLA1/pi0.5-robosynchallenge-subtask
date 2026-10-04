#!/usr/bin/env bash
# 실제 train.py 로 짧게 돌려 본다. 데이터·모델만 부르는 테스트가 못 잡는 것
# (첫 배치 이미지 로깅, optimizer 초기화, 체크포인트 저장)을 여기서 본다.
#
#   bash smoke_train.sh <config> <steps> [추가 env...]
set -uo pipefail
P=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
CONFIG=${1:?config}
STEPS=${2:-2}
EXP=${EXP:-smoke_$(date +%H%M%S)}
OUT=${OUT:-/workspace/smoke}
mkdir -p "$OUT"

cd "$P" || exit 1
export HF_LEROBOT_HOME=$P/training_data
export OPENPI_DATA_HOME=/workspace/.cache/openpi
export HF_HOME=/workspace/hf-home
export WANDB_MODE=offline
export WANDB_DIR=$OUT/wandb
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PI05_REAL_ACTION_DIM=${PI05_REAL_ACTION_DIM:-14}

echo "=== $CONFIG · $STEPS step · exp=$EXP ==="
echo "    SUBTASK_W=${PI05_SUBTASK_W:-0} MISTAKE=${PI05_MISTAKE:-0} MEM=${PI05_MEM_FRAMES:-1} batch=${SMOKE_BATCH:-config}"

ARGS=(scripts/train.py "$CONFIG" --exp-name "$EXP" --overwrite
      --num-train-steps "$STEPS" --checkpoint-base-dir "$OUT/ckpt")
[ -n "${SMOKE_BATCH:-}" ] && ARGS+=(--batch-size "$SMOKE_BATCH")

./.venv/bin/python "${ARGS[@]}" 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "train.py rc=$rc"
[ "$rc" -eq 0 ] && echo "SMOKE-OK $CONFIG" || echo "SMOKE-FAIL $CONFIG"
exit "$rc"
