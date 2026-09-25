#!/usr/bin/env bash
# 디코딩이 끝나면 학습 -> 평가 적용을 이어서 돌린다.
# stdout 은 Monitor 가 읽는 이벤트 스트림이므로 요약만 내보내고 자세한 로그는 파일로 남긴다.
set -u
cd /root/rsc_recover/subtask

D=/workspace/subtask
NDEMO=${NDEMO:-150}
NEVAL=${NEVAL:-102}

echo "대기: 디코딩 완료를 기다린다 (데모 $NDEMO, 평가 $NEVAL)"
for _ in $(seq 1 240); do
  d=$(ls "$D/frames_demo"/*.npy 2>/dev/null | wc -l)
  e=$(ls "$D/frames_eval"/*.npy 2>/dev/null | wc -l)
  [ "$d" -ge "$NDEMO" ] && [ "$e" -ge "$NEVAL" ] && break
  sleep 20
done
d=$(ls "$D/frames_demo"/*.npy 2>/dev/null | wc -l)
e=$(ls "$D/frames_eval"/*.npy 2>/dev/null | wc -l)
echo "디코딩 확보: 데모 $d · 평가 $e"
if [ "$d" -lt 60 ]; then echo "FAILED: 데모 에피소드가 $d 개뿐 — 중단"; exit 1; fi

echo "학습 시작"
python3 train_phase.py \
  --frames "$D/frames_demo" \
  --schedule /root/rsc_recover/ds_audit/schedule_sim.json \
  --task items_handover \
  --val-episodes 30 --epochs 3 --batch 96 \
  --out "$D/phase_resnet18.pt" > "$D/train.log" 2>&1
rc=$?
grep -a "검증 정확도" "$D/train.log"
if [ $rc -ne 0 ]; then echo "FAILED: 학습 (rc=$rc)"; tail -5 "$D/train.log"; exit 1; fi
echo "학습 완료"

echo "평가 롤아웃 적용 시작"
python3 apply_eval.py \
  --model "$D/phase_resnet18.pt" \
  --eval-frames "$D/frames_eval" \
  --demo-frames "$D/frames_demo" \
  --n-demo 10 \
  --out /root/rsc_recover/subtask/eval_phase > "$D/apply.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "FAILED: 평가 적용 (rc=$rc)"; tail -5 "$D/apply.log"; exit 1; fi
sed -n '/^요약/,$p' "$D/apply.log"
echo "PIPELINE-DONE"
