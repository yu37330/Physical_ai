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
# 採点はL4 24GB。A100やT4で測った値はそのまま使えないので記録して警告する。
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "$GPU_NAME" != *"L4"* ]]; then
  echo "WARNING: 採点環境はNVIDIA L4 24GBです。現在は $GPU_NAME のため、"
  echo "         VRAMとLatencyは参考値になります。最終判定はL4で取り直してください。"
fi

if [[ ! -d "$MODEL_TARGET" ]] || [[ -z "$(ls -A "$MODEL_TARGET" 2>/dev/null)" ]]; then
  echo "Checkpoint not found: $MODEL_TARGET" >&2
  echo "Run colab_setup.sh, then place the checkpoint there. For a base-weights" >&2
  echo "measurement run (not submittable), copy \$BASE_CHECKPOINT into it." >&2
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
