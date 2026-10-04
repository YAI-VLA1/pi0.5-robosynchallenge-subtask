#!/usr/bin/env bash
# 빈 /workspace 에서 subtask 실험 학습 재개까지 한 번에. 멱등하다.
#
#   bash /root/rsc_recover/bootstrap_subtask.sh
#
# 무엇이 다른가 (bootstrap_handover.sh 와 비교)
#   같은 환경(venv·데이터·norm_stats)을 쓰되, pi0.5 에 subtask 문장 생성을 붙이는
#   패치 3개를 더 얹고 새 config 로 돈다. 자세한 것은 subtask/findings_step2.md.
set -uo pipefail

REC=/root/rsc_recover
SUB=$REC/subtask
WS=/workspace/rsc_ws
R=$WS/RoboSynChallenge
PI05=$R/policy/pi05
CONFIG=pi05_robosyn_items_handover_subtask_cos82k
BASE_CONFIG=pi05_robosyn_items_handover_lora_cos82k
EXP=items_handover_subtask_cos82k
TARGET_STEP=81999
REPO=yai-robosync/pi05-items-handover-subtask-cos82k
CKPT_BASE=/workspace/rsc_ckpt
CKPT_ROOT=$CKPT_BASE/$CONFIG/$EXP
SCHED=$REC/ds_audit/schedule_sim.json
LOG=$REC/bootstrap_subtask.log

exec >> "$LOG" 2>&1
echo "================ $(date -Is) bootstrap_subtask 시작 ================"
step() { echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

export HF_TOKEN="$(tr -d '\n' < /root/.cache/huggingface/token)"

# ── 1. 환경 (venv · 데이터 · norm_stats) ────────────────────────────────
# bootstrap_subtask_env.sh 는 bootstrap_handover.sh 의 1~6 절과 같다(멱등).
if [ ! -x "$PI05/.venv/bin/python" ] || ! "$PI05/.venv/bin/python" -c "import openpi" 2>/dev/null \
   || [ ! -f "$PI05/assets/$BASE_CONFIG/RoboSynChallenge/cobotmagic_Sim_items_handover/norm_stats.json" ]; then
  step "환경 구축 (bootstrap_subtask_env.sh — venv·데이터 8.5GB·norm_stats)"
  bash "$REC/bootstrap_subtask_env.sh" || exit 1
fi
step "환경 확인"
"$PI05/.venv/bin/python" -c "import openpi" || exit 1

# ── 2. subtask 패치 3종 ─────────────────────────────────────────────────
# 전부 멱등이고 --check 를 지원한다. 앵커가 안 맞으면 즉시 멈춘다
# (업스트림이 바뀐 것이므로 조용히 넘어가면 안 된다).
step "subtask 패치 적용"
# ★ apply_loss_mask 는 bootstrap_subtask_env.sh 의 clone 블록 안에만 있었다.
#   저장소가 이미 있는 경로로 재구축하면 통째로 빠지는데 **증상이 없다**.
#   여기서 매번 부른다 (멱등). 2026-10-04 에 실제로 당했다.
python3 "$REC/apply_loss_mask.py" "$R" || exit 1
python3 "$SUB/apply_subtask_patch.py" "$R" || exit 1        # tokenizer/transform/policy/pi0/gemma
# 순서 중요: subtask -> decode -> v2. 이 순서로만 검증했다 (2026-09-26).
python3 "$SUB/apply_decode_patch.py" "$R" || exit 1         # 추론 디코드 (평가에 필요)
python3 "$SUB/apply_subtask_v2.py" "$R" || exit 1           # Task 제거·EOS·정확도·분리로깅
python3 "$SUB/prepare_subtask_config.py" "$PI05" "$SCHED" || exit 1   # 데이터 배선 + TrainConfig

# 하드 체크. 하나라도 빠지면 학습을 띄우지 않는다.
step "패치 검증"
python3 "$REC/verify_patches.py" "$R" --subtask || exit 1

# ── 3. norm_stats 를 새 config 자산 경로로 ──────────────────────────────
NS_OLD=$PI05/assets/$BASE_CONFIG/RoboSynChallenge
NS_NEW=$PI05/assets/$CONFIG
if [ ! -f "$NS_NEW/RoboSynChallenge/cobotmagic_Sim_items_handover/norm_stats.json" ]; then
  step "norm_stats 복사 ($BASE_CONFIG -> $CONFIG)"
  mkdir -p "$NS_NEW" && cp -r "$NS_OLD" "$NS_NEW/" || exit 1
fi

# ── 4. 이어서 하기: HF 에 체크포인트가 있으면 받는다 ────────────────────
if ! ls -1 "$CKPT_ROOT" 2>/dev/null | grep -qE '^[0-9]+$'; then
  step "HF 에서 체크포인트 조회 ($REPO)"
  # 별도 파일로 뺐다. 여기에 heredoc 으로 끼워 넣었더니 시스템 python3 이 불려
  # huggingface_hub 이 없었고, 실패해도 넘어가 step 0 부터 다시 돌았다 (2026-09-26).
  "$PI05/.venv/bin/python" "$REC/fetch_subtask_ckpt.py" "$REPO" "$CKPT_ROOT" || exit 1
fi

# ── 5. wandb run id 복원 ────────────────────────────────────────────────
# train.py 는 --resume 시 <ckpt_dir>/wandb_id.txt 를 **반드시** 읽는다. 없으면 죽는다.
# 이 파일은 /workspace 에 있어 파드가 죽으면 사라지므로 /root 에 사본을 둔다.
# 둘 다 없는데 체크포인트가 있으면(= 이번처럼 사본을 못 남긴 채 파드가 죽은 경우)
# 새 id 를 만들어서라도 진행한다 — 곡선이 끊길 뿐 학습은 살려야 한다.
WID=$REC/wandb_id_$EXP.txt
if [ -f "$WID" ] && [ ! -f "$CKPT_ROOT/wandb_id.txt" ]; then
  mkdir -p "$CKPT_ROOT" && cp "$WID" "$CKPT_ROOT/wandb_id.txt"
  step "wandb run id 복원 ($(cat "$WID"))"
elif [ -f "$CKPT_ROOT/wandb_id.txt" ] && [ ! -f "$WID" ]; then
  cp "$CKPT_ROOT/wandb_id.txt" "$WID"
  step "wandb run id 를 /root 에 백업 ($(cat "$WID"))"
elif [ ! -f "$CKPT_ROOT/wandb_id.txt" ] && ls -1 "$CKPT_ROOT" 2>/dev/null | grep -qE '^[0-9]+$'; then
  NEWID=$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')
  mkdir -p "$CKPT_ROOT" && printf '%s' "$NEWID" > "$CKPT_ROOT/wandb_id.txt"
  printf '%s' "$NEWID" > "$WID"
  step "wandb run id 가 없다 — 새로 만든다 ($NEWID). 기존 곡선과는 끊긴다."
fi

# ── 6. 학습 + 업로더 ────────────────────────────────────────────────────
# PI05_SUBTASK_W=0 이면 subtask 가 완전히 꺼진다(원래 pi0.5 동작).
# wandb 팀 YAI-VLA 는 API 키 권한이 없어 개인 엔티티로 돈다 (baseline 도 거기다).
if ! ps -eo cmd | grep -q "[t]rain.py $CONFIG"; then
  step "학습 기동"
  setsid nohup env CONFIG="$CONFIG" EXP="$EXP" TARGET_STEP="$TARGET_STEP" \
    WANDB_ENTITY_OVERRIDE=robosyn333-yai \
    PI05_SUBTASK_W="${PI05_SUBTASK_W:-1.0}" PI05_REAL_ACTION_DIM=14 \
    bash "$REC/run_training.sh" > "$REC/train_subtask_wrap.log" 2>&1 < /dev/null &
fi
sleep 3
if ! ps -eo cmd | grep -q "[w]atchdog.sh"; then
  step "업로더 워치독 기동 (공개 저장소, 10,000 배수 보존)"
  setsid nohup env CKPT_ROOT="$CKPT_ROOT" HF_REPO_ID="$REPO" FINAL_STEP="$TARGET_STEP" \
    MILESTONE=10000 HF_REPO_PRIVATE=0 TAG=subtask \
    bash "$REC/watchdog.sh" > "$REC/watchdog_subtask_wrap.log" 2>&1 < /dev/null &
fi

step "완료 — 학습 로그: $CKPT_BASE/train_$EXP.log"
