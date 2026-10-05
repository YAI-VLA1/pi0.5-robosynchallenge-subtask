#!/usr/bin/env bash
# 설정 조합별 loss. 가장 낮은 조합이 학습 설정이다.
P=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
CK=${1:?params 경로}
cd $P || exit 1
run() { # run <config> <w> <mistake> <mem>
  env PYTHONPATH=src HF_LEROBOT_HOME=$P/training_data OPENPI_DATA_HOME=/workspace/.cache/openpi \
    HF_HOME=/workspace/hf-home XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PI05_REAL_ACTION_DIM=14 PI05_SUBTASK_W=$2 PI05_MISTAKE=$3 PI05_MEM_FRAMES=$4 \
    PROBE_CFG=$1 PROBE_CKPT=$CK \
    ./.venv/bin/python -u /root/rsc_recover/probe_ckpt_config.py 2>&1 \
    | grep -aE "^PROBE|Error|error|OOM|RESOURCE" | head -3
}
S=pi05_robosyn_items_handover_subtask_cos82k
SM=pi05_robosyn_items_handover_subtask_mistake_cos82k
B=pi05_robosyn_items_handover_lora_cos82k
run $SM 1.0 1 6      # 저장소 이름이 가리키는 조합
run $SM 1.0 1 1
run $S  1.0 0 6
run $S  1.0 0 1
run $B  0.0 0 1      # 대조군
echo "PROBE-ALL-DONE"
