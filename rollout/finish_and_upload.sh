#!/usr/bin/env bash
# 평가가 끝나면 지표·라벨을 뽑고 HuggingFace 에 올린다.
#
#   bash finish_and_upload.sh <롤아웃 데이터셋> <HF 하위경로>
#
# 왜 자동화하나
#   2026-10-04 에 baseline 100 에피소드(약 4시간 수집)를 /workspace 가 날아가며
#   통째로 잃었다. 수집이 끝나는 **즉시** 올려야 한다.
set -uo pipefail
REC=/root/rsc_recover
P=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
D=${1:?롤아웃 데이터셋 경로}
NAME=${2:-$(basename "$D")}
REPO=${REPO:-yai-robosync/handover-rollouts}
step(){ echo ">>> [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

step "평가 종료 대기"
for i in $(seq 1 960); do
  pgrep -f "[e]val_policy.py" >/dev/null || break
  sleep 30
done
sleep 20

NPQ=$(find "$D/data" -name '*.parquet' 2>/dev/null | wc -l)
NMP=$(find "$D/videos" -name '*.mp4' 2>/dev/null | wc -l)
step "수집 결과 parquet $NPQ · mp4 $NMP"
[ "$NPQ" -ge 1 ] || { echo "★ parquet 가 없다"; exit 1; }

step "단계별 지표"
"$P/.venv/bin/python" "$REC/stage_metrics.py" "$D" 2>&1 | tail -14

step "mistake 라벨 (정렬은 녹화 단계에서 이미 맞다 — fix_rollout_alignment 불필요)"
"$P/.venv/bin/python" "$REC/mistake_labels.py" "$D" 2>&1 | tail -12

step "README"
cat > "$D/README.md" <<EOF
# $NAME

pi0.5 롤아웃 — items_handover

| | |
|---|---|
| 체크포인트 | ${CKPT_DESC:-mem_ep1 step 10661 (H100 batch 32, 1 epoch)} |
| 설정 | subtask + mistake + MEM(T=6) |
| 에피소드 | $NPQ |
| 형식 | LeRobot v2.1 |

## 들어 있는 것

observation.state / observation.qvel / observation.qf / action (각 14축),
pen_pose / holder_pose (4x4), 3뷰 영상(cam_high, cam_right_wrist, cam_left_wrist).

## (obs, action) 정렬

**이 데이터는 정렬이 맞다.** 녹화기가 step **직전** 관측과 그 action 을 짝짓는다
(RSC_ALIGN 패치). 2026-10-04 이전에 수집한 \`subtask/\` 는 step **이후** 관측과
짝지어져 한 칸 어긋나 있으니 \`rollout/fix_rollout_alignment.py\` 를 거쳐야 한다.

검증: (action-state)·(다음 state-state) 코사인 중앙값이 대본 데모(+0.918)와
비슷하면 정렬이 맞다.

## 곁들인 파일

- \`rollout_meta.json\` — 에피소드별 seed / success
- \`stage_metrics.json\` — 단계별 도달 지표
- \`mistake_labels.json\` — 실수 시점 / 절단 지점 (학습용 라벨)
EOF

step "HuggingFace 업로드 -> $REPO/$NAME"
"$P/.venv/bin/python" "$REC/backup_rollouts.py" "$D" "$REPO" "$NAME" || exit 1
step "완료"
echo "UPLOAD-DONE $REPO/$NAME"
