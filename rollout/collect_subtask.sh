#!/usr/bin/env bash
# subtask 체크포인트로 롤아웃 100 에피소드를 모으고 **즉시 HF 에 백업**한다.
# (2026-10-04: baseline 100개를 /workspace 소실로 잃었다. 백업은 선택이 아니다.)
set -uo pipefail
R=/workspace/rsc_ws/RoboSynChallenge; PI05=$R/policy/pi05; SUB=/root/rsc_recover/subtask
step(){ echo "######## [$(TZ=Asia/Seoul date +%H:%M)] $*"; }

# subtask 패치가 필요하다 (프롬프트·decode). 순서 중요: loss_mask -> subtask -> decode -> v2
if ! grep -q "token_loss_mask" "$PI05/src/openpi/models/pi0.py"; then
  step "subtask 패치"
  python3 /root/rsc_recover/apply_loss_mask.py "$R" || exit 1
  python3 "$SUB/apply_subtask_patch.py" "$R" || exit 1
  python3 "$SUB/apply_decode_patch.py" "$R" || exit 1
  python3 "$SUB/apply_subtask_v2.py" "$R" || exit 1
fi
grep -q "pi05_robosyn_items_handover_subtask_cos82k" "$PI05/src/openpi/training/config.py" || {
  step "subtask TrainConfig"
  python3 "$SUB/prepare_subtask_config.py" "$PI05" /root/rsc_recover/ds_audit/schedule_sim.json || exit 1; }
# 롤아웃 기록 패치
grep -q "RolloutRecorder" "$R/scripts/eval_policy.py" || {
  step "롤아웃 기록 패치"
  python3 /root/rsc_recover/apply_rollout_record2.py "$R" || exit 1; }

sed 's|^CONFIG=pi05_robosyn_items_handover_lora_cos82k|CONFIG=pi05_robosyn_items_handover_subtask_cos82k|' \
  /root/rsc_recover/record_failures.sh > /root/rsc_recover/record_failures_subtask.sh

step "subtask 100 에피소드 수집"
EXP=rollout_subtask SRC=/workspace/ckpt_dl/subtask/checkpoints/81999 \
  OUT=/workspace/rollout_ds/subtask PI05_SUBTASK_W=1.0 \
  bash /root/rsc_recover/record_failures_subtask.sh 100

step "HF 백업"
"$PI05/.venv/bin/python" /root/rsc_recover/backup_rollouts.py \
  /workspace/rollout_ds/subtask yai-robosync/handover-rollouts subtask || echo "★ 백업 실패"
step "COLLECT-SUBTASK-DONE"
