#!/usr/bin/env bash
# 빈 머신 -> 학습 가능 상태. 멱등하다.
#
#   export WS=$HOME/rsc_ws          # 작업 루트
#   export HF_TOKEN=hf_...
#   bash setup_train.sh [--subtask] [--mistake] [--mem] [--all]
#
# 선택 환경변수
#   NORM_FROM_CKPT=<ckpt>   그 체크포인트의 assets/ 에서 norm_stats 를 가져온다 (권장).
#                           없으면 compute_norm_stats.py 를 돌리는데 **영상을 디코딩해
#                           몇 시간** 걸린다. 경고하고 계속할지 묻는다.
set -euo pipefail

REC=$(cd "$(dirname "$0")/.." && pwd)
WS=${WS:?WS 를 지정할 것 (예: export WS=$HOME/rsc_ws)}
R=$WS/RoboSynChallenge
PI05=$R/policy/pi05
DATA=$PI05/training_data/RoboSynChallenge/cobotmagic_Sim_items_handover
BASE_CONFIG=pi05_robosyn_items_handover_lora_cos82k
COMMIT=9815e9e
ASSET_ID=RoboSynChallenge/cobotmagic_Sim_items_handover

OPTS=(); CONFIGS=("$BASE_CONFIG")
for a in "$@"; do case "$a" in
  --subtask) OPTS+=(--subtask); CONFIGS+=(pi05_robosyn_items_handover_subtask_cos82k) ;;
  --mistake) OPTS+=(--mistake); CONFIGS+=(pi05_robosyn_items_handover_mistake_cos82k
                                          pi05_robosyn_items_handover_subtask_mistake_cos82k) ;;
  --mem)     OPTS+=(--mem) ;;
  --all)     OPTS+=(--all);     CONFIGS+=(pi05_robosyn_items_handover_subtask_cos82k
                                          pi05_robosyn_items_handover_mistake_cos82k
                                          pi05_robosyn_items_handover_subtask_mistake_cos82k) ;;
  *) echo "모르는 인자: $a"; exit 1 ;;
esac; done
step() { echo ">>> [$(date +%H:%M:%S)] $*"; }

# 디렉터리가 없을 때 find 가 1 을 내고 pipefail 에 걸려 셸이 죽는다 (리뷰 R4).
count_files() {   # count_files <디렉터리> <패턴>
  [ -d "$1" ] || { echo 0; return 0; }
  find "$1" -name "$2" -type f | wc -l
}

# ── 1. 시스템 패키지 ────────────────────────────────────────────────────
command -v ffmpeg >/dev/null || { step "ffmpeg 설치"; \
  (sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg build-essential pkg-config) \
  || { echo "★ ffmpeg 를 직접 설치할 것 (없으면 torchcodec 이 libavutil.so.56 을 못 찾는다)"; exit 1; }; }
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
  python3 "$REC/setup/prepare_repo.py" "$PI05" > /dev/null
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

# ── 4. 소스 패치 (의존 순서는 apply_patches.sh 가 안다) ──────────────────
# ★ clone 여부와 무관하게 매번 돈다. 예전에 clone 블록 안에 뒀다가 저장소가
#   이미 있는 경로로 재구축했을 때 손실 마스킹이 통째로 빠졌다 (리뷰 R3 와 같은 뿌리).
step "패치 적용"
bash "$REC/setup/apply_patches.sh" "$R" "${OPTS[@]:-}"

# ── 5. 데이터 (8.5GB) ───────────────────────────────────────────────────
NPQ=$(count_files "$DATA/data" '*.parquet')
NMP4=$(count_files "$DATA/videos" '*.mp4')
if [ "$NPQ" -lt 1000 ] || [ "$NMP4" -lt 3000 ]; then
  step "데이터 회수 (8.5GB, 약 12분) — 현재 parquet $NPQ/1000, mp4 $NMP4/3000"
  "$PI05/.venv/bin/python" "$REC/setup/fetch_data.py" "$DATA"
fi
step "데이터 확인 parquet $(count_files "$DATA/data" '*.parquet')/1000 · mp4 $(count_files "$DATA/videos" '*.mp4')/3000"

# ── 6. norm_stats ───────────────────────────────────────────────────────
# 선택한 config 전부에 통계를 놓는다. 예전 setup 은 baseline 에만 만들어서
# subtask/mistake config 로 데이터로더를 만들면 통계를 못 찾았다 (리뷰 R6).
NS_SRC=""
if [ -n "${NORM_FROM_CKPT:-}" ]; then
  NS_SRC="$NORM_FROM_CKPT/assets/$ASSET_ID/norm_stats.json"
  [ -f "$NS_SRC" ] || { echo "★ $NS_SRC 가 없다"; exit 1; }
  step "norm_stats 를 체크포인트에서 가져온다 ($NORM_FROM_CKPT)"
else
  NS_BASE=$PI05/assets/$BASE_CONFIG/$ASSET_ID/norm_stats.json
  if [ ! -f "$NS_BASE" ]; then
    cat <<EOF
★ norm_stats 가 없고 NORM_FROM_CKPT 도 안 줬다.
  compute_norm_stats.py 는 영상을 디코딩해 **수 시간** 걸린다.
  기존 체크포인트에서 이어 학습할 거라면 그 체크포인트의 통계를 쓰는 편이
  정확하고 즉시 끝난다:
      NORM_FROM_CKPT=/path/to/checkpoints/81999 bash setup_train.sh ...
  그래도 새로 계산하려면 CONFIRM_SLOW_NORM=1 을 주고 다시 실행할 것.
EOF
    [ "${CONFIRM_SLOW_NORM:-0}" = 1 ] || exit 1
    step "norm_stats 계산 (영상 디코딩 — 수 시간)"
    ( cd "$PI05" && HF_LEROBOT_HOME="$PI05/training_data" OPENPI_DATA_HOME="$WS/.cache/openpi" \
      .venv/bin/python scripts/compute_norm_stats.py --config-name "$BASE_CONFIG" )
  fi
  NS_SRC=$NS_BASE
fi
for C in $(printf '%s\n' "${CONFIGS[@]}" | sort -u); do
  D=$PI05/assets/$C/$ASSET_ID
  mkdir -p "$D"
  cp "$NS_SRC" "$D/norm_stats.json"
done
step "norm_stats 배치 완료: $(printf '%s ' $(printf '%s\n' "${CONFIGS[@]}" | sort -u))"

# ── 7. 하드 체크 ────────────────────────────────────────────────────────
step "패치 검증"
VA=""
printf '%s\n' "${OPTS[@]:-}" | grep -q -- --all && VA="--all"
[ -z "$VA" ] && for o in "${OPTS[@]:-}"; do VA="$VA $o"; done
python3 "$REC/patches/verify_patches.py" "$R" $VA

step "config 로드 검증 (문자열 검사만으로는 깨진 config 를 못 잡는다 — 리뷰 지적)"
( cd "$PI05" && env PYTHONPATH=src HF_LEROBOT_HOME="$PI05/training_data" \
    OPENPI_DATA_HOME="$WS/.cache/openpi" PI05_MISTAKE=1 PI05_SUBTASK_W=1.0 \
    .venv/bin/python "$REC/patches/verify_configs.py" $(printf '%s\n' "${CONFIGS[@]}" | sort -u) )

cat <<EOF

SETUP-DONE
  WS   = $WS
  PI05 = $PI05
  다음: WS=$WS CONFIG=<config> bash $REC/setup/train.sh
EOF
