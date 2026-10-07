#!/usr/bin/env bash
# CobotMagic URDF (순기구학용). 메시는 필요 없다 — pinocchio 의 운동학 모델만 쓴다.
set -euo pipefail
D=/workspace/.cache/embodichain_data
[ -f "$D/extract/CobotMagicWithGripperV100.urdf" ] && { echo "URDF 이미 있음"; exit 0; }
mkdir -p "$D/extract"
/root/hfenv/bin/python - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download("DexForceAI/embodichain_data", "robot_assets/CobotMagicArmV3.zip",
                      repo_type="dataset", local_dir="$D/dl"))
PY
unzip -q -o "$D/dl/robot_assets/CobotMagicArmV3.zip" -d "$D/extract/"
# 압축은 extract/ 바로 아래로 풀린다 (wrist_bbox.py 후보에 그 경로를 넣어 뒀다).
ls -l "$D/extract/CobotMagicWithGripperV100.urdf"
echo URDF-DONE
