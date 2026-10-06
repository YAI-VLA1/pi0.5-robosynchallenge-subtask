#!/usr/bin/env bash
# 소스 패치 전부를 **의존 순서대로** 적용한다. 멱등하다.
#
#   bash apply_patches.sh <repo 루트> [--subtask] [--mistake] [--mem] [--recording]
#
#   --recording 은 평가 중 롤아웃을 저장할 때만 필요하다 (학습에는 불필요).
#
# 순서가 왜 이런가 (리뷰 R3 — 실제로 깨졌던 순서를 고친 것)
#   prepare_repo       cos40k TrainConfig 를 넣는다. prepare_handover 의 삽입 기준점이다.
#   prepare_handover   LeRobotEmbodiChainDataConfig 에 state_key/image_key_* 를 넣는다.
#                      MEM wire2 의 앵커("observation/state": self.state_key)가 여기서 생긴다.
#   subtask 3종        tokenizer/transforms/policy/pi0/gemma/config
#   prepare_subtask    repack 에 frame_index 를 넣는다. apply_mistake 의 앵커다.
#   apply_mistake      repack 에 episode_index, 프롬프트에 Mistake 필드
#   mem                wire2 가 repack 에 *_is_pad 를 넣는다
#   prepare_mistake    병합 데이터셋 TrainConfig (데이터셋이 있어야 한다)
set -euo pipefail

REC=$(cd "$(dirname "$0")/.." && pwd)
R=${1:?사용법: apply_patches.sh <repo 루트> [--subtask] [--mistake] [--mem]}
shift || true
P=$R/policy/pi05
SCHED=${SCHED:-$REC/schedule/schedule_sim.json}

SUB=0; MIS=0; MEM=0; REC_ROLL=0; BBOX=0
for a in "$@"; do case "$a" in
  --subtask) SUB=1 ;; --mistake) MIS=1 ;; --mem) MEM=1 ;;
  --recording) REC_ROLL=1 ;;
  --bbox) BBOX=1; MIS=1 ;;
  --all) SUB=1; MIS=1; MEM=1; BBOX=1 ;;
  "") ;;                       # 빈 인자는 무시 (빈 배열 전개 방어)
  *) echo "모르는 인자: $a"; exit 1 ;;
esac; done
# mistake 구현은 subtask 가 만든 frame_index 앵커를 쓴다. 혼자서는 못 돈다.
[ "$MIS" = 1 ] && SUB=1
say() { echo "── $*"; }

say "prepare_repo (cos40k TrainConfig + pyproject)"
python3 "$REC/setup/prepare_repo.py" "$P" > /dev/null

say "apply_loss_mask (패딩 18축 제외)"
python3 "$REC/patches/apply_loss_mask.py" "$R" > /dev/null

say "prepare_handover (컬럼 파라미터화 + baseline TrainConfig)"
python3 "$REC/setup/prepare_handover.py" "$P" > /dev/null

if [ "$SUB" = 1 ]; then
  say "subtask 3종"
  python3 "$REC/subtask/apply_subtask_patch.py" "$R" > /dev/null
  python3 "$REC/subtask/apply_decode_patch.py"  "$R" > /dev/null
  python3 "$REC/subtask/apply_subtask_v2.py"    "$R" > /dev/null
  say "apply_decode_eos (EOS 뒤를 pad 로 — 리뷰 R2)"
  python3 "$REC/subtask/apply_decode_eos.py" "$P" > /dev/null
  say "prepare_subtask_config (frame_index + subtask TrainConfig)"
  python3 "$REC/subtask/prepare_subtask_config.py" "$P" "$SCHED" > /dev/null
fi

# 롤아웃 녹화. --recording 을 줬을 때만 설치한다.
#   align 패치의 앵커는 record2 가 만든 코드다. 깨끗한 upstream 에는 그 코드가
#   없어서, 조건을 "eval_policy.py 가 있으면" 으로 두면 fresh 설치가 여기서
#   exit 1 로 멈춘다 (리뷰 N2). 학습만 할 때는 녹화가 필요 없다.
if [ "$REC_ROLL" = 1 ]; then
  say "apply_rollout_record2 (롤아웃 녹화)"
  python3 "$REC/rollout/apply_rollout_record2.py" "$R" > /dev/null
  say "apply_rollout_align (녹화 (o_t, a_t) 정렬 — record2 뒤에 와야 한다)"
  python3 "$REC/rollout/apply_rollout_align.py" "$R" > /dev/null
elif grep -q "_roll" "$R/scripts/eval_policy.py" 2>/dev/null; then
  # 이미 녹화가 깔린 checkout 이면 정렬만 맞춘다 (기존 머신 복구 경로).
  say "apply_rollout_align (기존 녹화 코드에 정렬만)"
  python3 "$REC/rollout/apply_rollout_align.py" "$R" > /dev/null
fi

if [ "$MIS" = 1 ]; then
  say "apply_mistake (episode_index + Mistake 필드)"
  python3 "$REC/mistake/apply_mistake.py" "$P" > /dev/null
fi

if [ "$BBOX" = 1 ]; then
  say "bbox (Pen 슬롯 + 두 슬롯 디코드 + CE 인덱스 수정)"
  python3 "$REC/bbox/apply_bbox.py"        "$P" > /dev/null
  python3 "$REC/bbox/apply_bbox_decode.py" "$P" > /dev/null
  python3 "$REC/bbox/apply_bbox_fix.py"    "$P" > /dev/null
  # PI05_BBOX=1 이면 cam_high 기하 증강을 끈다 — 라벨은 원본 좌표라
  # 크롭 32px + 회전 35px 가 state-only 기준선(26.2px)을 넘는 잡음이 된다.
  python3 "$REC/bbox/apply_bbox_aug.py"    "$P" > /dev/null
  # "박스 없음" 을 CE 에서 빼지 않고 <loc0000>×4 로 가르친다.
  python3 "$REC/bbox/apply_bbox_nobox.py"  "$P" > /dev/null
  # 손목 카메라 슬롯 (PI05_BBOX_WRIST=1 로 켠다)
  python3 "$REC/bbox/apply_bbox_wrist.py"  "$P" > /dev/null
  # 집기 구간 손실 가중 (PI05_GRASP_W=3.0 로 켠다)
  python3 "$REC/bbox/apply_grasp_weight.py" "$P" > /dev/null
fi

if [ "$MEM" = 1 ]; then
  say "MEM"
  bash "$REC/mem/apply_all.sh" "$P" > /dev/null
fi

# 병합 데이터셋이 있을 때만. 없으면 조용히 건너뛰지 않고 이유를 찍는다.
MIX=$P/training_data/RoboSynChallenge/cobotmagic_Sim_items_handover_mix/meta/mistake_starts.json
if [ "$MIS" = 1 ]; then
  if [ -f "$MIX" ]; then
    say "prepare_mistake_config (병합 TrainConfig)"
    python3 "$REC/mistake/prepare_mistake_config.py" "$P" > /dev/null
  else
    echo "   (병합 데이터셋이 아직 없다 — mistake/merge_mistake_ds.py 뒤에 다시 돌릴 것)"
  fi
fi

say "문법 확인"
"$P/.venv/bin/python" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" \
  "$P/src/openpi/training/config.py" 2>/dev/null \
  || python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$P/src/openpi/training/config.py"

echo "APPLY-PATCHES-DONE"
