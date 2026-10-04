#!/usr/bin/env bash
# baseline 체크포인트로 롤아웃을 돌리며 state/action 을 LeRobot 포맷으로 저장한다.
set -uo pipefail
R=/workspace/rsc_ws/RoboSynChallenge; PI05=$R/policy/pi05
CONFIG=pi05_robosyn_items_handover_lora_cos82k
EXP=${EXP:-items_handover_record}
STEP=81999; N=${1:-2}
SRC=${SRC:-/workspace/ckpt_dl/baseline/checkpoints/$STEP}
step(){ echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

CK=/workspace/rsc_ckpt/$CONFIG/$EXP; mkdir -p "$CK"
[ -d "$SRC/params" ] || { echo "체크포인트 없음: $SRC"; exit 1; }
ln -sfn "$SRC" "$CK/$STEP"
for d in /workspace/rsc_ckpt/*/; do ln -sfn "${d%/}" "$PI05/checkpoints/$(basename "$d")"; done
DS=RoboSynChallenge/cobotmagic_Sim_items_handover
mkdir -p "$PI05/assets/$CONFIG/$DS" && cp "$SRC/assets/$DS/norm_stats.json" "$PI05/assets/$CONFIG/$DS/" || exit 1
step "체크포인트·norm_stats 준비 ($SRC)"

OUT=${OUT:-/workspace/rollout_ds/$EXP}
step "저장 경로 $OUT (시작 전 $(ls "$OUT/data" 2>/dev/null | wc -l) chunk)"
cd "$R" || exit 1
env PI05_RECORD_ROLLOUT="$OUT" PI05_RECORD_SRC="$EXP" \
    PI05_RECORD_REPO=yai-robosync/handover-rollouts PI05_REAL_ACTION_DIM=14 \
    HF_LEROBOT_HOME="$PI05/training_data" OPENPI_DATA_HOME=/workspace/.cache/openpi \
    HF_HOME=/workspace/hf-home \
  bash policy/pi05/eval.sh items_handover random "$CONFIG" "$EXP" 0 \
    --max_episodes "$N" --checkpoint_id "$STEP" > /workspace/record_$EXP.log 2>&1
echo "EXIT rc=$?"
tr '\r' '\n' < /workspace/record_$EXP.log | sed 's/\x1b\[[0-9;]*m//g' | grep -a "success rate" | tail -1
step "저장 결과"
find "$OUT" -maxdepth 3 -type d 2>/dev/null | head -10
find "$OUT" -name "*.parquet" 2>/dev/null | wc -l
find "$OUT" -name "*.mp4" 2>/dev/null | wc -l
[ -f "$OUT/meta/info.json" ] && python3 -c "
import json; d=json.load(open('$OUT/meta/info.json'))
print('  에피소드', d.get('total_episodes'), '프레임', d.get('total_frames'))
print('  features:', list(d.get('features',{}).keys()))"
echo "RECORD-DONE"
