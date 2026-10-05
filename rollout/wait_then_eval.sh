#!/usr/bin/env bash
# 10661 다운로드가 **완료 마커를 찍고 크기까지 맞을 때만** 평가를 띄운다.
# 디렉터리 존재로 판정하면 안 된다 — hf_hub_download 가 빈 디렉터리를 먼저 만든다
# (2026-10-05 에 그걸로 빈 체크포인트에 평가를 띄웠다).
set -u
for i in $(seq 1 480); do
  grep -aq "FETCH-DONE" /workspace/fetch_10661.log 2>/dev/null && break
  pgrep -f "[f]etch_mem_ep1.py" >/dev/null || { echo "★ 다운로드가 완료 없이 끝났다"; exit 1; }
  sleep 30
done
grep -aq "FETCH-DONE" /workspace/fetch_10661.log || { echo "★ 시간 초과"; exit 1; }
SZ=$(du -sb /workspace/mem_ep1/checkpoints/10661 | cut -f1)
[ "$SZ" -gt 6000000000 ] || { echo "★ 크기가 작다 ($SZ B)"; exit 1; }
echo ">>> 다운로드 확인 $(numfmt --to=iec "$SZ")"
CFG=pi05_robosyn_items_handover_subtask_mistake_cos82k
ln -sfn /workspace/mem_ep1/checkpoints/10661 "/workspace/rsc_ckpt/$CFG/mem_ep1/10661"
rm -rf /workspace/rollout_ds/mem_ep1_10661
RECORD=/workspace/rollout_ds/mem_ep1_10661 bash /root/rsc_recover/run_eval_mem_ep1.sh 100 10661
