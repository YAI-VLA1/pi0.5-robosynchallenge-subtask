#!/usr/bin/env bash
# 실제 train.py 로 (config, batch, T, CE) 조합별 3스텝을 돌려 본다.
# 통과/OOM 과 활성화 메모리를 남긴다. 이것이 production 상한의 근거다.
G=/root/rsc_recover
run() {  # run <태그> <config> <batch> <T> <subtask_w> <mistake>
  local tag=$1 cfg=$2 b=$3 t=$4 w=$5 m=$6
  echo "######## $tag  batch=$b T=$t SUBTASK_W=$w MISTAKE=$m"
  PI05_SUBTASK_W=$w PI05_MISTAKE=$m PI05_MEM_FRAMES=$t \
  EXP="smoke_$tag" SMOKE_BATCH=$b OUT=/workspace/smoke \
    bash $G/smoke_train.sh "$cfg" 3 2>&1 \
    | grep -aE "only reduced to|RESOURCE_EXHAUSTED|SMOKE-|train.py rc=|loss=|Error|error" \
    | sed 's/.*only reduced to /  활성화 /; s/ (.*originally//' | head -6
  echo
}
SUB=pi05_robosyn_items_handover_subtask_cos82k_post12k
BASE=pi05_robosyn_items_handover_lora_cos82k

run b1_t1_ce     $SUB  1 1 1.0 0
run b1_t1_noce   $BASE 1 1 0.0 0
run b2_t1_noce   $BASE 2 1 0.0 0
run b1_t6_ce     $SUB  1 6 1.0 0
echo "MATRIX-DONE"
