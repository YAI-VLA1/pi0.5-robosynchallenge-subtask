#!/usr/bin/env bash
# 빈 /workspace 에서 bbox+손목+집기가중 학습 재개까지 한 번에. 멱등하다.
#
#   bash /root/rsc_recover/bootstrap_bbox.sh
#
# 무엇을 학습하나 (2026-10-07)
#   MEM(6프레임) + mistake + cam_high bbox 에 더해 **손목 bbox 슬롯**,
#   **'박스 없음' CE**, **집기 구간 손실 가중(x3)** 이 들어간 런이다.
#   10,661 스텝(H100·배치 32)짜리 bbox 체크포인트에서 **가중치만** 물려받아
#   짧게 더 돈다 — 여기(3090)는 배치 2 라 처음부터는 현실적이지 않다.
#
# 왜 이 파일이 /root 에 있나
#   파드가 유휴로 정지되면 /workspace 가 통째로 비워진다. /root 는 살아남는다.
#   다시 올라온 뒤 이 스크립트 하나로 복구된다.
set -uo pipefail

REC=/root/rsc_recover
GH=$REC/gh_subtask
WS=/workspace/rsc_ws
R=$WS/RoboSynChallenge
PI05=$R/policy/pi05
SEED_ROOT=/workspace/rsc_seed
SEED=$SEED_ROOT/checkpoints/10661
SEED_REPO=yai-robosync/pi05-items-handover-subtask-mistake-mem-bbox-ep1

BASE_CFG=pi05_robosyn_items_handover_subtask_mistake_cos82k
STEPS=${POST_STEPS:-12000}
EXP=${EXP:-items_handover_bbox_wrist_grasp}
REPO=${HF_REPO_ID:-yai-robosync/pi05-items-handover-bbox-wrist-grasp}
CKPT_BASE=/workspace/rsc_ckpt
TAG=bboxwrist
LOG=$REC/bootstrap_bbox.log

CONFIG="${BASE_CFG}_post$((STEPS / 1000))k"
TARGET_STEP=$((STEPS - 1))
CKPT_ROOT=$CKPT_BASE/$CONFIG/$EXP

exec >> "$LOG" 2>&1
echo "================ $(date -Is) bootstrap_bbox 시작 ================"
step() { echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

# 중복 기동 방지
exec 9>"$REC/.lock_bbox"
flock -n 9 || { echo "다른 bootstrap_bbox 가 진행 중이다. 종료."; exit 0; }
if ps -eo cmd | grep -q "[t]rain.py $CONFIG"; then
  step "학습이 이미 돌고 있다. 워치독만 확인한다."
fi

set +u; source /root/.bashrc >/dev/null 2>&1; set -u
export HF_TOKEN="$(tr -d '\n' < /root/.cache/huggingface/token)"
PY=$PI05/.venv/bin/python
HFPY=/root/hfenv/bin/python          # venv 가 아직 없을 때 쓰는 보조 파이썬

# ── 1. warm-start 체크포인트 + obb 라벨 ─────────────────────────────────
if [ ! -d "$SEED/params" ]; then
  step "warm-start 체크포인트 10661 + obb 라벨 (약 9GB)"
  "$HFPY" "$REC/bbox_run/fetch_seed.py" "$SEED_ROOT" || exit 1
fi

# ── 2. 환경 (clone · venv · 패치 · 데이터 · norm_stats) ─────────────────
step "setup_train.sh --all (멱등)"
WS=$WS HF_TOKEN=$HF_TOKEN NORM_FROM_CKPT=$SEED bash "$GH/setup/setup_train.sh" --all || exit 1

# ── 3. 실패 롤아웃 98 + 병합 데이터셋 ───────────────────────────────────
if [ ! -d /workspace/rollout_ds/subtask ]; then
  step "실패 롤아웃 내려받기"
  "$HFPY" "$REC/bbox_run/fetch_rollouts.py" || exit 1
fi
# mistake_labels.json 은 HF 업로드에 안 들어 있다 (롤아웃 원본만 올렸다).
# 결정적으로 다시 만들어진다 — 그리퍼 신호와 펜 높이에서 계산한다.
if [ ! -f /workspace/rollout_ds/subtask/mistake_labels.json ]; then
  step "실수 라벨 재생성 (t_fail · 절단 지점)"
  "$PY" "$GH/mistake/mistake_labels.py" /workspace/rollout_ds/subtask || exit 1
fi
MIX=$PI05/training_data/RoboSynChallenge/cobotmagic_Sim_items_handover_mix
if [ ! -f "$MIX/meta/mistake_starts.json" ]; then
  step "병합 데이터셋 생성 (데모 1,000 + 실패 98)"
  PI05=$PI05 ROLLOUTS=/workspace/rollout_ds/subtask "$PY" "$GH/mistake/merge_mistake_ds.py" || exit 1
fi

# ── 4. bbox 라벨 두 벌 ──────────────────────────────────────────────────
# cam_high 는 obb(투영 GT)에서 바로 나온다. 손목은 순기구학이라 URDF 가 필요하다.
if [ ! -f /workspace/bbox_tokens/ep0999.npy ]; then
  step "cam_high bbox 토큰 (obb -> 4토큰)"
  "$PY" "$GH/bbox/make_bbox_labels.py" --src /workspace/obb_labels --out /workspace/bbox_tokens || exit 1
fi
if [ ! -f /workspace/wrist_bbox_tokens/ep0999.npy ]; then
  step "로봇 URDF (순기구학용)"
  bash "$REC/bbox_run/fetch_urdf.sh" || exit 1
  # pinocchio 는 학습 venv 에 넣지 않는다 — 라벨을 뽑는 데만 쓰는 의존이고,
  # 학습 환경을 건드릴 이유가 없다. 별도 venv 를 /workspace 에 만든다.
  FK=/workspace/fkenv/bin/python
  if [ ! -x "$FK" ]; then
    step "FK 전용 venv (pinocchio)"
    uv venv --python 3.11 /workspace/fkenv -q || exit 1
    uv pip install -q --python "$FK" pin numpy pandas pyarrow || exit 1
  fi
  step "손목 bbox 토큰 (FK 투영)"
  "$FK" "$GH/bbox/wrist_bbox.py" --extract /workspace/wrist_bbox_tokens || exit 1
fi

# ── 5. config 두 개 ─────────────────────────────────────────────────────
# (a) mistake config — bbox/손목/집기 인자를 리터럴로 박는다
step "prepare_mistake_config (bbox·손목·집기 인자)"
"$PY" "$GH/mistake/prepare_mistake_config.py" "$PI05" || exit 1
# (b) 거기서 가중치만 물려받는 post-train config
if ! grep -q "name=\"$CONFIG\"" "$PI05/src/openpi/training/config.py"; then
  step "prepare_posttrain_config ($BASE_CFG -> $CONFIG, $STEPS 스텝)"
  "$PY" "$GH/mistake/prepare_posttrain_config.py" "$PI05" "$BASE_CFG" "$SEED" "$STEPS" || exit 1
fi

# ── 6. 이 런의 최신 체크포인트를 HF 에서 ────────────────────────────────
mkdir -p "$CKPT_ROOT"
if ! ls -1 "$CKPT_ROOT" 2>/dev/null | grep -qE '^[0-9]+$'; then
  step "HF 에서 이 런의 체크포인트 조회 ($REPO)"
  "$PY" "$REC/bbox_run/fetch_run_ckpt.py" "$REPO" "$CKPT_ROOT" || exit 1
fi
RESUME=$(ls -1 "$CKPT_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
step "재개 지점: ${RESUME:-없음 (10661 가중치에서 0부터)}"

# wandb run id — --resume 은 이 파일이 없으면 죽는다
WID=$REC/wandb_id_$EXP.txt
if [ -f "$WID" ] && [ ! -f "$CKPT_ROOT/wandb_id.txt" ]; then
  cp "$WID" "$CKPT_ROOT/wandb_id.txt"; step "wandb id 복원 ($(cat "$WID"))"
elif [ -f "$CKPT_ROOT/wandb_id.txt" ] && [ ! -f "$WID" ]; then
  cp "$CKPT_ROOT/wandb_id.txt" "$WID"; step "wandb id 를 /root 에 백업"
elif [ ! -f "$CKPT_ROOT/wandb_id.txt" ] && [ -n "${RESUME:-}" ]; then
  NEWID=$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')
  printf '%s' "$NEWID" > "$CKPT_ROOT/wandb_id.txt"; printf '%s' "$NEWID" > "$WID"
  step "wandb id 가 없어 새로 만든다 ($NEWID) — 곡선만 끊긴다"
fi

[ "${SKIP_TRAIN:-0}" = 1 ] && { step "SKIP_TRAIN=1 — 기동하지 않는다"; exit 0; }

# ── 7. 학습 + 업로더 워치독 ─────────────────────────────────────────────
# ★ PI05_* 는 **학습과 추론에 같은 값**을 줘야 한다. 프롬프트 형식이 달라지면
#   조용히 안 죽고 loss 만 튄다.
if ! ps -eo cmd | grep -q "[t]rain.py $CONFIG"; then
  step "학습 기동 ($CONFIG · $EXP)"
  setsid nohup env CONFIG="$CONFIG" EXP="$EXP" TARGET_STEP="$TARGET_STEP" CKPT_BASE="$CKPT_BASE" \
    WANDB_ENTITY_OVERRIDE=robosyn333-yai \
    PI05_SUBTASK_W=1.0 PI05_MISTAKE=1 PI05_REAL_ACTION_DIM=14 \
    PI05_MEM_FRAMES=6 PI05_MEM_STRIDE_S=1.0 PI05_MEM_HIST_DROPOUT=0.3 \
    PI05_BBOX=1 PI05_BBOX_WRIST=1 PI05_GRASP_W=3.0 \
    bash "$REC/run_training.sh" > "$REC/train_${TAG}_wrap.log" 2>&1 < /dev/null &
  sleep 5
fi
# 워치독 생존은 PID 파일로 본다 — pgrep -f 는 진단 명령줄까지 잡는다.
WPID=$REC/watchdog_$TAG.pid
if ! { [ -f "$WPID" ] && kill -0 "$(cat "$WPID")" 2>/dev/null; }; then
  step "업로더 워치독 기동 (1,000 배수 보존, 공개)"
  rm -f "$REC/uploader_$TAG.pid"
  setsid nohup env CKPT_ROOT="$CKPT_ROOT" HF_REPO_ID="$REPO" FINAL_STEP="$TARGET_STEP" \
    MILESTONE=2000 HF_REPO_PRIVATE=0 TAG=$TAG \
    bash "$REC/watchdog.sh" > "$REC/watchdog_${TAG}_wrap.log" 2>&1 < /dev/null &
  echo $! > "$WPID"
fi

step "완료 — 학습 로그: $CKPT_BASE/train_$EXP.log"
echo "================ $(date -Is) bootstrap_bbox 끝 ================"
