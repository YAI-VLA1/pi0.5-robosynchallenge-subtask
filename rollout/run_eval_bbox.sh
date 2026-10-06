#!/usr/bin/env bash
# bbox 런 체크포인트 평가.
#
#   RECORD=<경로> bash run_eval_bbox.sh [에피소드수] [step]
#
# ★ 학습과 같은 환경변수를 줘야 한다. 프롬프트 형식이 어긋나면 조용히 안 죽고
#   성능만 무너진다. bbox 는 PI05_BBOX=1 이 학습·추론 양쪽에 있어야 한다.
set -uo pipefail
REC=/root/rsc_recover
WS=/workspace/rsc_ws; R=$WS/RoboSynChallenge; PI05=$R/policy/pi05
N=${1:-10}; STEP=${2:-10661}
# _mix 데이터셋이 없으면 mistake config 가 등록되지 않는다. 추론에는 상관없다 —
# InjectBBox/InjectMistake 는 repack(학습 전용)이고, Pen 슬롯은 PI05_BBOX 로 켜진다.
CONFIG=${CONFIG:-pi05_robosyn_items_handover_subtask_cos82k}
EXP=bbox_ep1
LOG=/workspace/eval_bbox_${STEP}_n${N}.log
step(){ echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

CK=/workspace/rsc_ckpt/$CONFIG/$EXP
[ -d "$CK/$STEP/params" ] || { echo "★ $CK/$STEP/params 가 없다"; exit 1; }
mkdir -p "$PI05/checkpoints"
for d in /workspace/rsc_ckpt/*/; do ln -sfn "${d%/}" "$PI05/checkpoints/$(basename "$d")"; done
grep -qE "^headless: true" "$PI05/deploy_policy.yml" || \
  sed -i 's/^headless:.*/headless: true/' "$PI05/deploy_policy.yml"

step "패치 확인"
python3 "$REC/verify_patches.py" "$R" --subtask --mistake --mem --bbox | tail -4 || exit 1

DUMP=/workspace/dump_bbox_${STEP}_n${N}.jsonl
rm -f "$DUMP"
python3 "$REC/subtask/apply_dump_subtask.py" "$R" >/dev/null || exit 1

REC_ARG=""; [ -n "${RECORD:-}" ] && REC_ARG="$RECORD"
step "평가 $N 에피소드 · step $STEP (로그 $LOG)"
cd "$R" || exit 1
env PI05_SUBTASK_W=1.0 PI05_MISTAKE=1 PI05_BBOX=1 PI05_REAL_ACTION_DIM=14 \
    PI05_MEM_FRAMES=6 PI05_MEM_STRIDE_S=1.0 \
    ${REC_ARG:+PI05_RECORD_ROLLOUT="$REC_ARG"} \
    PI05_DUMP_SUBTASK="$DUMP" \
    HF_LEROBOT_HOME="$PI05/training_data" \
    OPENPI_DATA_HOME=/workspace/.cache/openpi \
    HF_HOME=/workspace/hf-home \
  bash policy/pi05/eval.sh items_handover random "$CONFIG" "$EXP" 0 \
    --max_episodes "$N" --checkpoint_id "$STEP" > "$LOG" 2>&1
rc=$?
echo "EVAL-EXIT rc=$rc $(date -Is)" | tee -a "$LOG"
echo "덤프 $(wc -l < "$DUMP" 2>/dev/null || echo 0) 호출 -> $DUMP"
