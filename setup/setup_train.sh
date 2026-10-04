#!/usr/bin/env bash
# 빈 머신 -> 학습 가능 상태. 멱등하다. 약 25분.
#
#   export WS=$HOME/rsc_ws          # 작업 루트
#   export HF_TOKEN=hf_...          # 데이터 접근
#   bash setup/setup_train.sh [--subtask] [--mistake] [--mem]
#
# VESSL 전용 가정이 없다. 경로는 전부 $WS 아래다.
set -euo pipefail

REC=$(cd "$(dirname "$0")/.." && pwd)
WS=${WS:?WS 를 지정할 것 (예: export WS=$HOME/rsc_ws)}
R=$WS/RoboSynChallenge
PI05=$R/policy/pi05
DATA=$PI05/training_data/RoboSynChallenge/cobotmagic_Sim_items_handover
BASE_CONFIG=pi05_robosyn_items_handover_lora_cos82k
COMMIT=9815e9e

WANT_SUBTASK=0; WANT_MISTAKE=0; WANT_MEM=0
for a in "$@"; do
  case "$a" in
    --subtask) WANT_SUBTASK=1 ;;
    --mistake) WANT_MISTAKE=1 ;;
    --mem)     WANT_MEM=1 ;;
    --all)     WANT_SUBTASK=1; WANT_MISTAKE=1; WANT_MEM=1 ;;
    *) echo "모르는 인자: $a"; exit 1 ;;
  esac
done
step() { echo ">>> [$(date +%H:%M:%S)] $*"; }

# ── 1. 시스템 패키지 ────────────────────────────────────────────────────
# ffmpeg 가 없으면 torchcodec 이 libavutil.so.56 을 못 찾는다.
command -v ffmpeg >/dev/null || { step "ffmpeg 설치"; \
  (sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg build-essential pkg-config) \
  || { echo "★ ffmpeg 를 직접 설치할 것"; exit 1; }; }
command -v uv >/dev/null || { echo "★ uv 가 없다: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# ── 2. 저장소 ───────────────────────────────────────────────────────────
mkdir -p "$WS"
if [ ! -d "$R" ]; then
  step "RoboSynChallenge 클론 ($COMMIT)"
  git -C "$WS" clone -q https://github.com/EDEM-AI/RoboSynChallenge.git
  git -C "$R" checkout -q "$COMMIT"
  git -C "$R" apply "$REC/setup/robosync_fixes.patch"
fi

# ── 3. venv ─────────────────────────────────────────────────────────────
if [ ! -x "$PI05/.venv/bin/python" ] || ! "$PI05/.venv/bin/python" -c "import openpi" 2>/dev/null; then
  step "venv 구축 (약 6분)"
  python3 "$REC/setup/prepare_repo.py" "$PI05"
  ( cd "$PI05" && GIT_LFS_SKIP_SMUDGE=1 uv sync )
  cp -r "$PI05/src/openpi/models_pytorch/transformers_replace/"* \
        "$PI05/.venv/lib/python3.11/site-packages/transformers/"
fi
"$PI05/.venv/bin/python" -c "import hf_xet" 2>/dev/null || \
  uv pip install -q --python "$PI05/.venv/bin/python" hf_xet
step "openpi 확인"
"$PI05/.venv/bin/python" -c "
import lerobot, openpi.training.data_loader as _, jax
print('    lerobot', lerobot.__version__, '· 장치', jax.devices())"

# ── 4. 소스 패치 ────────────────────────────────────────────────────────
# ★ clone 여부와 무관하게 **매번** 돈다. 예전에 clone 블록 안에 뒀다가
#   저장소가 이미 있는 경로로 재구축했을 때 손실 마스킹이 통째로 빠졌다.
step "패치 적용"
python3 "$REC/patches/apply_loss_mask.py" "$R"
VERIFY_ARGS=""
if [ "$WANT_SUBTASK" = 1 ]; then
  python3 "$REC/subtask/apply_subtask_patch.py" "$R"
  python3 "$REC/subtask/apply_decode_patch.py"  "$R"
  python3 "$REC/subtask/apply_subtask_v2.py"    "$R"
  VERIFY_ARGS="$VERIFY_ARGS --subtask"
fi
if [ "$WANT_MISTAKE" = 1 ]; then
  python3 "$REC/mistake/apply_mistake.py" "$PI05"
  VERIFY_ARGS="$VERIFY_ARGS --mistake"
fi
if [ "$WANT_MEM" = 1 ]; then
  bash "$REC/mem/apply_all.sh" "$PI05"
  VERIFY_ARGS="$VERIFY_ARGS --mem"
fi

# ── 5. 학습 설정 ────────────────────────────────────────────────────────
step "TrainConfig 삽입"
python3 "$REC/setup/prepare_handover.py" "$PI05"
[ "$WANT_SUBTASK" = 1 ] && python3 "$REC/subtask/prepare_subtask_config.py" \
  "$PI05" "$REC/schedule/schedule_sim.json"

# ── 6. 데이터 (8.5GB) ───────────────────────────────────────────────────
NPQ=$(find "$DATA/data" -name '*.parquet' 2>/dev/null | wc -l)
NMP4=$(find "$DATA/videos" -name '*.mp4' 2>/dev/null | wc -l)
if [ "$NPQ" -lt 1000 ] || [ "$NMP4" -lt 3000 ]; then
  step "데이터 회수 (8.5GB, 약 12분) — 현재 parquet $NPQ/1000, mp4 $NMP4/3000"
  "$PI05/.venv/bin/python" "$REC/setup/fetch_data.py" "$DATA"
fi

# ── 7. norm_stats ───────────────────────────────────────────────────────
# 일반 경로는 영상까지 디코딩해 3시간 18분. --fast-fix 는 parquet 만 읽어 2분.
NS=$PI05/assets/$BASE_CONFIG/RoboSynChallenge/cobotmagic_Sim_items_handover/norm_stats.json
if [ ! -f "$NS" ]; then
  step "norm_stats 계산 (약 2분)"
  ( cd "$PI05" && HF_LEROBOT_HOME="$PI05/training_data" OPENPI_DATA_HOME="$WS/.cache/openpi" \
    .venv/bin/python scripts/compute_norm_stats.py --config-name "$BASE_CONFIG" --fast-fix )
fi

# ── 8. 하드 체크 ────────────────────────────────────────────────────────
step "패치 검증"
python3 "$REC/patches/verify_patches.py" "$R" $VERIFY_ARGS

cat <<EOF

SETUP-DONE
  WS   = $WS
  PI05 = $PI05
  다음: bash $REC/setup/train.sh
EOF
