#!/usr/bin/env bash
# 학습 실행. 환경변수로 실험 조합을 고른다.
#
#   export WS=$HOME/rsc_ws
#   CONFIG=pi05_robosyn_items_handover_subtask_mistake_cos82k \
#   PI05_SUBTASK_W=1.0 PI05_MISTAKE=1 PI05_MEM_FRAMES=6 \
#   bash setup/train.sh
#
# ★ 여기 준 값은 **추론에도 똑같이** 줘야 한다. 프롬프트 형식이 달라지면
#   조용히 안 죽고 loss 만 튄다 (2026-09-26: 0.015 -> 10.25).
set -euo pipefail

WS=${WS:?WS 를 지정할 것}
PI05=$WS/RoboSynChallenge/policy/pi05
CONFIG=${CONFIG:-pi05_robosyn_items_handover_lora_cos82k}
EXP=${EXP:-$CONFIG}
RESUME=${RESUME:-0}

export HF_LEROBOT_HOME=$PI05/training_data
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$WS/.cache/openpi}
export PI05_REAL_ACTION_DIM=${PI05_REAL_ACTION_DIM:-14}
export PI05_SUBTASK_W=${PI05_SUBTASK_W:-0.0}
export PI05_MISTAKE=${PI05_MISTAKE:-0}
export PI05_MEM_FRAMES=${PI05_MEM_FRAMES:-1}
export PI05_MEM_STRIDE_S=${PI05_MEM_STRIDE_S:-1.0}
export PI05_MEM_HIST_DROPOUT=${PI05_MEM_HIST_DROPOUT:-0.3}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export PYTHONUNBUFFERED=1

cat <<EOF
config  $CONFIG
exp     $EXP
subtask CE 가중치 $PI05_SUBTASK_W · mistake $PI05_MISTAKE · MEM 프레임 $PI05_MEM_FRAMES
EOF

cd "$PI05"
if [ "$RESUME" = 1 ]; then
  exec .venv/bin/python scripts/train.py "$CONFIG" --exp-name "$EXP" --resume
fi
exec .venv/bin/python scripts/train.py "$CONFIG" --exp-name "$EXP" --overwrite
