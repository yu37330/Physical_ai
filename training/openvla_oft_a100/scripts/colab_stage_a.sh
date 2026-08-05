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
colab::notify_on_exit "Stage A $STAGE ($MAX_STEPS steps)"

RLDS_BUILDER_DIR="$RLDS_ROOT/$DATASET_NAME/$DATASET_VERSION"
if [[ ! -f "$RLDS_BUILDER_DIR/dataset_info.json" ]]; then
  echo "RLDS builder directory not found: $RLDS_BUILDER_DIR" >&2
  echo "DATASET_PROFILE is '$DATASET_PROFILE'." >&2
  # The default is mini, so a converted full dataset is easy to miss. Naming what
  # exists beats making the caller guess, and beats picking one automatically:
  # training on the wrong dataset would not announce itself.
  available=()
  for candidate in "$WORK_ROOT"/rlds/*/; do
    [[ -f "$candidate$DATASET_NAME/$DATASET_VERSION/dataset_info.json" ]] || continue
    available+=("$(basename "$candidate")")
  done
  if (( ${#available[@]} )); then
    echo "Converted profiles present: ${available[*]}" >&2
    echo "Re-run with DATASET_PROFILE=${available[0]}." >&2
    exit 1
  fi

  # /content is wiped with the VM, and the dataset also gets deleted by hand to
  # make room for the submission archive. Either way the copy on Drive is what
  # saves the hour of reconversion, so say it is there rather than sending the
  # caller back to a script that would re-download 2,400 files.
  restorable=()
  for candidate in "$DRIVE_RLDS_ROOT"/*/; do
    [[ -f "$candidate$DATASET_NAME/$DATASET_VERSION/dataset_info.json" ]] || continue
    restorable+=("$(basename "$candidate")")
  done
  if (( ${#restorable[@]} )); then
    echo "On Drive and restorable: ${restorable[*]}" >&2
    echo "Restore it, then re-run this script:" >&2
    echo "  DATASET_PROFILE=${restorable[0]} bash $SCRIPT_DIR/colab_dataset_prepare.sh" >&2
    echo "  DATASET_PROFILE=${restorable[0]} bash $0 $STAGE" >&2
  else
    echo "Run colab_dataset_prepare.sh first." >&2
  fi
  exit 1
fi
if [[ ! -d "$BASE_CHECKPOINT" ]]; then
  echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
  echo "Run colab_setup.sh first." >&2
  exit 1
fi

S1_RUN_DIR="$RUN_ROOT/stage_a_s1_head_proprio_100"
# The gate asks whether S1 passed, not whether this particular VM ran it. Every
# run persists to Drive, so a fresh runtime would otherwise be told to redo six
# minutes of training whose result is already sitting there.
if [[ "$STAGE" == "s2" ]] \
  && ! compgen -G "$S1_RUN_DIR/*/action_head--*checkpoint.pt" > /dev/null \
  && compgen -G "$DRIVE_EXPERIMENTS/stage_a_s1_head_proprio_100/*/action_head--*checkpoint.pt" > /dev/null
then
  colab::section "Restoring the S1 run from Drive"
  mkdir -p "$RUN_ROOT"
  cp -R "$DRIVE_EXPERIMENTS/stage_a_s1_head_proprio_100" "$S1_RUN_DIR"
  echo "Restored: $S1_RUN_DIR"
fi
if [[ "$STAGE" == "s2" ]] && ! compgen -G "$S1_RUN_DIR/*/action_head--*checkpoint.pt" > /dev/null; then
  echo "S1 produced no action head checkpoint under $S1_RUN_DIR." >&2
  echo "S2 must not run until S1 passes its gate." >&2
  exit 1
fi

# Last, because it imports torch and is slower than the checks above.
#
# colab_action_parity.sh and colab_submission_viability.sh both replace the
# OpenVLA-OFT transformers fork with the PyPI build. Training on the PyPI build
# would silently use causal attention while the submission runtime applies the
# bidirectional patch, so the model would be trained under different semantics
# than it is evaluated with. Nothing would error; only the score would suffer.
if ! python training/openvla_oft_a100/scripts/check_openvla_env.py --require-parc-dataset; then
  echo >&2
  echo "The OpenVLA-OFT environment is not ready for training." >&2
  echo "Run colab_setup.sh to restore it, then re-run this script." >&2
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
