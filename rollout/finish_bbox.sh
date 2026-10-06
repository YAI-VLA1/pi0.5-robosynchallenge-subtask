#!/usr/bin/env bash
# bbox 평가 마무리: 지표 -> 라벨 -> bbox 정확도 -> 영상 -> HF 업로드
set -uo pipefail
REC=/root/rsc_recover
P=/workspace/rsc_ws/RoboSynChallenge/policy/pi05
D=/workspace/rollout_ds/bbox_ep1_10661
DUMP=/workspace/dump_bbox_10661_n100.jsonl
VID=/workspace/bbox_vid_10661
step(){ echo; echo "########## [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

for i in $(seq 1 240); do pgrep -f "[e]val_policy.py" >/dev/null || break; sleep 30; done
sleep 15

step "단계별 지표"
"$P/.venv/bin/python" "$REC/stage_metrics.py" "$D" 2>&1 | sed -n '1,14p'

step "mistake 라벨"
"$P/.venv/bin/python" "$REC/mistake_labels.py" "$D" 2>&1 | tail -12

step "bbox 정확도 (★ 지름길 검증)"
cd "$P" && env PYTHONPATH=src "$P/.venv/bin/python" "$REC/bbox_accuracy.py" "$D" "$DUMP" 2>&1 | tail -22

step "영상 (bbox + subtask 오버레이)"
rm -rf "$VID"
cd "$P" && env PYTHONPATH=src HF_HOME=/workspace/hf-home \
  "$P/.venv/bin/python" "$REC/subtask_video.py" "$D" "$DUMP" "$VID" 2>&1 | tail -3
echo "영상 $(ls $VID/*.mp4 2>/dev/null | wc -l) 개 · $(du -sh $VID 2>/dev/null | cut -f1)"

step "README"
NPQ=$(find "$D/data" -name '*.parquet' | wc -l)
cat > "$D/README.md" <<EOF
# bbox_ep1_10661

pi0.5 롤아웃 — items_handover · **subtask + mistake + MEM(T=6) + bbox**

| | |
|---|---|
| 체크포인트 | \`yai-robosync/pi05-items-handover-subtask-mistake-mem-bbox-ep1\` step 10661 |
| 학습 | H100 batch 32, 1 epoch |
| 에피소드 | $NPQ |
| 형식 | LeRobot v2.1 |

state/qvel/qf/action (각 14축), pen_pose/holder_pose (4x4), 3뷰 영상.
\`rollout_meta.json\`(seed/success) · \`stage_metrics.json\` · \`mistake_labels.json\` 동봉.

**(obs, action) 정렬이 맞다** — 녹화기가 step 직전 관측과 짝짓는다(RSC_ALIGN).
2026-10-04 이전 \`subtask/\` 는 어긋나 있어 \`rollout/fix_rollout_alignment.py\` 가 필요하다.

생성된 subtask 문장과 bbox 토큰은 \`dump_bbox_10661_n100.jsonl\` 에 있다
(한 줄 = 정책 호출 하나, Pen 4 | "; Subtask:" 4 | Subtask 14 토큰).
EOF
cp "$DUMP" "$D/" 2>/dev/null

step "HF 업로드 — 롤아웃"
"$P/.venv/bin/python" "$REC/backup_rollouts.py" "$D" yai-robosync/handover-rollouts bbox_ep1_10661 || exit 1

step "HF 업로드 — 영상"
"$P/.venv/bin/python" "$REC/upload_videos_hf.py" "$VID=videos/bbox_ep1_10661" || exit 1

echo; echo "FINISH-BBOX-DONE"
