#!/usr/bin/env bash
# 学習前のGo/No-Go: 提出物がL4 24GB・10秒/推論・120秒起動・20GB ZIPに収まるかを実測する。
#
#   bash training/openvla_oft_a100/scripts/colab_submission_viability.sh
#
# MODEL_STRATEGY.mdの実験順序#1にあたる。ここで落ちる構成は学習しても提出できないため、
# Stage A学習より先に回す。ColabのL4ランタイムで実行すること。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

SUBMISSION_DIR="$PROJECT_ROOT/submission/openvla_oft_offline"
MODEL_TARGET="$SUBMISSION_DIR/model_weights/openvla_oft_plus"
REPORT="${VIABILITY_REPORT:-$WORK_ROOT/submission_viability.json}"

colab::section "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"

# 7Bをbf16で載せるだけで約15GB要る。T4は14.5GBかつTuringでbf16非対応なので、
# 測定に入っても必ずOOMするだけ時間を捨てることになる。
if (( GPU_TOTAL_MIB < 20000 )) && [[ "${ALLOW_SMALL_GPU:-0}" != "1" ]]; then
  echo "This GPU has ${GPU_TOTAL_MIB}MiB, which cannot hold the 7B checkpoint." >&2
  echo "Scoring uses an NVIDIA L4 24GB; select an L4 (or A100) runtime." >&2
  echo "Set ALLOW_SMALL_GPU=1 only to measure a model that actually fits." >&2
  exit 1
fi
# 採点はL4 24GB。A100で測ったLatencyはそのまま使えないので警告だけ残す。
if [[ "$GPU_NAME" != *"L4"* ]]; then
  echo "WARNING: 採点環境はNVIDIA L4 24GBです。現在は $GPU_NAME のため、"
  echo "         VRAMとLatencyは参考値になります。最終判定はL4で取り直してください。"
fi

# 計測目的なら Base 重みをそのまま置いて構わない。ただしこの構成は「公開重みを
# 実質的に変更せず推論する」に該当するため提出できない（docs/OFFICIAL_RULES.md 6章）。
if [[ "${STAGE_BASE_CHECKPOINT:-0}" == "1" ]]; then
  colab::section "Staging base checkpoint for measurement only"
  if [[ ! -d "$BASE_CHECKPOINT" ]]; then
    echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
    echo "Run colab_setup.sh first." >&2
    exit 1
  fi
  mkdir -p "$MODEL_TARGET"
  rsync -a --delete "$BASE_CHECKPOINT/" "$MODEL_TARGET/"
  echo "NOTE: base weights only. Not submittable; measurement use only."
fi

if [[ ! -d "$MODEL_TARGET" ]] || [[ -z "$(ls -A "$MODEL_TARGET" 2>/dev/null)" ]]; then
  echo "Checkpoint not found: $MODEL_TARGET" >&2
  echo "Run colab_setup.sh, then either place the trained checkpoint there or" >&2
  echo "re-run with STAGE_BASE_CHECKPOINT=1 for a base-weights measurement." >&2
  exit 1
fi

colab::section "Submission dependencies"
python -m pip install -q -r "$SUBMISSION_DIR/requirements.txt"

colab::section "Measuring"
python scripts/measure_submission_viability.py \
  --submission-dir "$SUBMISSION_DIR" \
  --num-requests "${VIABILITY_REQUESTS:-30}" \
  --output "$REPORT"

colab::section "Persist to Drive"
if colab::require_drive; then
  colab::persist "$REPORT" "$DRIVE_EXPERIMENTS/submission_viability/$(basename "$REPORT")"
fi

colab::report_disk
colab::section "Viability measurement complete"
echo "Report: $REPORT"
