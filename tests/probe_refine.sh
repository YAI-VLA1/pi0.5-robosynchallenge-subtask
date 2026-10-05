#!/usr/bin/env bash
# 교란 제거판.
#  ① MISTAKE 는 **같은 _mix 데이터셋**에서 필드 유무만 바꿔 비교한다
#     (앞 실험은 mistake 켠 쪽만 실패 롤아웃이 섞인 데이터라 비교가 안 됐다)
#  ② MEM 프레임 수를 2/4/6/8 로 훑는다
P=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
CK=${1:?params 경로}
cd $P || exit 1
run() { # run <config> <w> <mistake> <mem>
  env PYTHONPATH=src HF_LEROBOT_HOME=$P/training_data OPENPI_DATA_HOME=/workspace/.cache/openpi \
    HF_HOME=/workspace/hf-home XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PI05_REAL_ACTION_DIM=14 PI05_SUBTASK_W=$2 PI05_MISTAKE=$3 PI05_MEM_FRAMES=$4 \
    PI05_MEM_STRIDE_S=1.0 PROBE_CFG=$1 PROBE_CKPT=$CK \
    ./.venv/bin/python -u /root/rsc_recover/probe_ckpt_config.py 2>&1 \
    | grep -aE "^PROBE|RESOURCE|Error" | head -2
}
SM=pi05_robosyn_items_handover_subtask_mistake_cos82k
echo "--- ① 같은 _mix 데이터 · Mistake 필드만 on/off ---"
run $SM 1.0 1 6
run $SM 1.0 0 6
echo "--- ② MEM 프레임 수 (mistake on) ---"
for T in 2 4 8; do run $SM 1.0 1 $T; done
echo "PROBE-REFINE-DONE"
