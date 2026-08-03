#!/usr/bin/env bash
# Notebook 03 equivalent: run one Stage A stage on /content/work and persist only
# the trained components to Drive.
#
#   bash training/openvla_oft_a100/scripts/colab_stage_a.sh s1   # 100 steps
#   bash training/openvla_oft_a100/scripts/colab_stage_a.sh s2   # 500 steps
#
# 原則: do not run s2 unless s1 produced a checkpoint. Stage A trains the action
# head and proprio projector only, so the run directory stays small.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

STAGE="${1:-}"
case "$STAGE" in
  s1) MAX_STEPS="${MAX_STEPS:-100}"; RUN_NAME="stage_a_s1_head_proprio_100" ;;
  s2) MAX_STEPS="${MAX_STEPS:-500}"; RUN_NAME="stage_a_s2_head_proprio_500" ;;
  *) echo "Usage: $0 {s1|s2}" >&2; exit 2 ;;
esac

RLDS_BUILDER_DIR="$RLDS_ROOT/$DATASET_NAME/1.0.0"
if [[ ! -f "$RLDS_BUILDER_DIR/dataset_info.json" ]]; then
  echo "RLDS builder directory not found: $RLDS_BUILDER_DIR" >&2
  echo "Run colab_dataset_prepare.sh first (DATASET_PROFILE=$DATASET_PROFILE)." >&2
  exit 1
fi
if [[ ! -d "$BASE_CHECKPOINT" ]]; then
  echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
  echo "Run colab_setup.sh first." >&2
  exit 1
fi

S1_RUN_DIR="$RUN_ROOT/stage_a_s1_head_proprio_100"
if [[ "$STAGE" == "s2" ]] && ! compgen -G "$S1_RUN_DIR/*/action_head--*checkpoint.pt" > /dev/null; then
  echo "S1 produced no action head checkpoint under $S1_RUN_DIR." >&2
  echo "S2 must not run until S1 passes its gate." >&2
  exit 1
fi

RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

colab::section "Stage A $STAGE ($MAX_STEPS steps)"
echo "Dataset:    $RLDS_ROOT ($DATASET_MIXTURE)"
echo "Checkpoint: $BASE_CHECKPOINT"
echo "Run dir:    $RUN_DIR"

# train_smoke.sh calls vla-scripts/finetune.py relative to the OpenVLA-OFT tree.
cd "$OPENVLA_ROOT"
DATA_ROOT_DIR="$RLDS_ROOT" \
DATASET_NAME="$DATASET_MIXTURE" \
CHECKPOINT_DIR="$BASE_CHECKPOINT" \
RUN_ROOT_DIR="$RUN_DIR" \
MAX_STEPS="$MAX_STEPS" \
USE_LORA="${USE_LORA:-False}" \
TRAIN_VLA_LORA="${TRAIN_VLA_LORA:-False}" \
  bash "$SCRIPT_DIR/train_smoke.sh"

cd "$PROJECT_ROOT"
colab::section "Persist trained components to Drive"
if colab::require_drive; then
  # Only the trained delta is durable. The base weights stay reproducible from
  # the pinned Hugging Face revision recorded in checkpoint_manifest.json.
  colab::persist "$RUN_DIR" "$DRIVE_EXPERIMENTS/$RUN_NAME" \
    "${STAGE_A_DRIVE_LIMIT_MB:-1024}"
fi

colab::report_disk
colab::section "Stage A $STAGE complete"
echo "Local run:  $RUN_DIR"
echo "Persisted:  $DRIVE_EXPERIMENTS/$RUN_NAME"
