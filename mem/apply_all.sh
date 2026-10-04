#!/usr/bin/env bash
# MEM 패치 전부. 인자는 policy/pi05 경로 (단, apply_mem_infer 는 저장소 루트를 받는다).
# 전부 멱등이다. 하나라도 실패하면 즉시 멈춘다.
set -euo pipefail
PI05=${1:?사용법: apply_all.sh <repo>/policy/pi05}
D=$(cd "$(dirname "$0")" && pwd)

for s in apply_mem.py \
         apply_mem_data.py \
         apply_mem_wire2.py \
         apply_mem_obs.py \
         apply_mem_numframes.py \
         apply_mem_resize.py \
         apply_mem_aug.py \
         apply_mem_crop.py \
         apply_mem_dropout.py; do
  echo "── $s"
  python3 "$D/$s" "$PI05"
done

# 추론 경로(deploy_policy.py / pi_model.py)는 저장소 루트 기준이다. 평가할 때만 필요하다.
if [ -f "$PI05/deploy_policy.py" ]; then
  echo "── apply_mem_infer.py (추론 프레임 버퍼)"
  python3 "$D/apply_mem_infer.py"  "$PI05"
  python3 "$D/apply_mem_obswin.py" "$PI05"
fi

echo "MEM-APPLY-ALL-DONE"
