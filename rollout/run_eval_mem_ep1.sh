#!/usr/bin/env bash
# mem-ep1 (H100 batch 32) 체크포인트 평가.
#
#   bash run_eval_mem_ep1.sh [에피소드수] [step]
#
# 설정은 loss 프로브로 역추적했다 (probe_all.sh / probe_refine.sh):
#   subtask 확실 (대조군 0.0385 vs 0.0061), MEM T≈6 확실 (T=1 0.0120 -> T=6 0.0061,
#   T=8 에서 포화). mistake 는 loss 로 판별 안 돼 저장소 이름을 따른다.
#
# ★ 학습과 같은 환경변수를 줘야 한다. 프롬프트 형식이 어긋나면 조용히 안 죽고
#   성능만 무너진다 (2026-09-26: 0.015 -> 10.25).
set -uo pipefail
REC=/root/rsc_recover
WS=/workspace/rsc_ws; R=$WS/RoboSynChallenge; PI05=$R/policy/pi05
N=${1:-10}; STEP=${2:-7000}
CONFIG=pi05_robosyn_items_handover_subtask_mistake_cos82k
EXP=mem_ep1
LOG=/workspace/eval_mem_ep1_${STEP}_n${N}.log
step() { echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

CK=/workspace/rsc_ckpt/$CONFIG/$EXP
[ -d "$CK/$STEP/params" ] || { echo "★ $CK/$STEP/params 가 없다"; exit 1; }
mkdir -p "$PI05/checkpoints"
for d in /workspace/rsc_ckpt/*/; do ln -sfn "${d%/}" "$PI05/checkpoints/$(basename "$d")"; done

grep -qE "^headless: true" "$PI05/deploy_policy.yml" || \
  sed -i 's/^headless:.*/headless: true/' "$PI05/deploy_policy.yml"

step "패치 확인 (MEM 추론 경로가 들어가 있어야 한다)"
python3 "$REC/verify_patches.py" "$R" --subtask --mistake --mem \
  | tail -8 | grep -E "OK|✗|VERIFY" || exit 1

DUMP=/workspace/subtask_gen_mem_ep1_${STEP}_n${N}.jsonl
rm -f "$DUMP"
python3 "$REC/subtask/apply_dump_subtask.py" "$R" >/dev/null || exit 1

step "평가 $N 에피소드 · step $STEP (로그 $LOG)"
cd "$R" || exit 1
# RECORD=<경로> 를 주면 롤아웃을 LeRobot 데이터셋으로 남긴다.
# 성공률이 1% 대라 그걸로는 비교가 안 된다 — 단계별 지표(펜 들림 등)가 필요하고,
# 그러려면 pen_pose/state/action 이 있는 parquet 이 있어야 한다.
REC_ARG=""
[ -n "${RECORD:-}" ] && REC_ARG="$RECORD"

env PI05_SUBTASK_W=1.0 PI05_MISTAKE=1 PI05_REAL_ACTION_DIM=14 \
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
grep -aE "Success|success|성공" "$LOG" | tail -5
