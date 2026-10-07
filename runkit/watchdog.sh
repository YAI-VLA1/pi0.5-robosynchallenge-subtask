#!/usr/bin/env bash
# 업로더가 죽으면 되살린다.
#
# 학습은 run_training.sh 가 감독하지만 업로더는 아무도 보지 않았다. 조용히 죽으면
# 학습만 계속 돌고 백업이 끊긴 채로 남는데, 파드가 내려간 뒤에야 알게 된다.
#
# 생존 확인에 pgrep -f 를 쓰지 않는다 — 그 문자열을 담은 다른 명령줄(진단 명령,
# 이 스크립트 자신)까지 매칭해서 죽은 업로더를 살아 있다고 오판한다. PID 파일로 본다.
set -uo pipefail
REC=/root/rsc_recover
PI05=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
# 런마다 다른 값은 환경변수로 받는다. 기본값은 click_bell 런이다.
CKPT_ROOT=${CKPT_ROOT:-/workspace/rsc_ckpt/pi05_robosyn_click_bell_lora_cos40k/click_bell_cos40k}
REPO=${HF_REPO_ID:-yai-robosync/pi05-click-bell-cos40k}
FINAL_STEP=${FINAL_STEP:-39999}
MILESTONE=${MILESTONE:-5000}
HF_REPO_PRIVATE=${HF_REPO_PRIVATE:-1}
TAG=${TAG:-click_bell}
PIDFILE="$REC/uploader_$TAG.pid"
UPLOAD_STATE="$REC/uploaded_steps_$TAG.json"

alive() {
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

while true; do
  if ! alive; then
    if grep -q "마지막 체크포인트까지 올렸다" "$REC/uploader_$TAG.log" 2>/dev/null; then
      echo "[$(date -Is)] $TAG 마지막까지 업로드됨. 워치독 종료." >> "$REC/watchdog_$TAG.log"; exit 0
    fi
    CKPT_ROOT="$CKPT_ROOT" HF_REPO_ID="$REPO" FINAL_STEP="$FINAL_STEP" \
    MILESTONE="$MILESTONE" HF_REPO_PRIVATE="$HF_REPO_PRIVATE" UPLOAD_STATE="$UPLOAD_STATE" \
      nohup "$PI05/.venv/bin/python" "$REC/upload_checkpoints.py" >> "$REC/uploader_$TAG.log" 2>&1 &
    echo "$!" > "$PIDFILE"
    echo "[$(date -Is)] $TAG 업로더 기동 (pid $!)" >> "$REC/watchdog_$TAG.log"
  fi
  sleep 60
done
