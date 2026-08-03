#!/usr/bin/env bash
# Notebook 02 equivalent: fetch the selected LeRobot episodes and convert them to
# RLDS under /content/work, leaving only selection JSON and reports on Drive.
#
#   bash training/openvla_oft_a100/scripts/colab_dataset_prepare.sh          # mini
#   DATASET_PROFILE=full bash .../colab_dataset_prepare.sh                   # 800 episodes
#
# 運用原則13: run the mini profile and all pre-gates before the full conversion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"
colab::require_drive

FULL_SELECTION="${FULL_SELECTION:-$DRIVE_DATASETS/libero_plus_selection_v001.json}"
if [[ ! -f "$FULL_SELECTION" ]]; then
  echo "Selection file not found: $FULL_SELECTION" >&2
  echo "Set FULL_SELECTION to the 800 episode selection JSON on Drive." >&2
  exit 1
fi

mkdir -p "$DRIVE_DATASETS" "$SOURCE_ROOT" "$RLDS_ROOT"

case "$DATASET_PROFILE" in
  mini)
    SELECTION_FILE="$DRIVE_DATASETS/mini_selection_v001.json"
    ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DRIVE_DATASETS/mini_e2e_v001}"
    PROMOTE_MANIFEST=0
    PARITY_EPISODES_PER_SPLIT="${PARITY_EPISODES_PER_SPLIT:-1}"

    colab::section "Mini selection (2 train / 1 validation)"
    python -m src.data.build_mini_selection \
      --selection "$FULL_SELECTION" \
      --train-count "${MINI_TRAIN_COUNT:-2}" \
      --validation-count "${MINI_VALIDATION_COUNT:-1}" \
      --output "$SELECTION_FILE"
    ;;
  full)
    SELECTION_FILE="$FULL_SELECTION"
    ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DRIVE_DATASETS/stage_a_balanced_v001}"
    PROMOTE_MANIFEST="${PROMOTE_MANIFEST:-1}"
    PARITY_EPISODES_PER_SPLIT="${PARITY_EPISODES_PER_SPLIT:-2}"

    if [[ "$PROMOTE_MANIFEST" == "1" && -z "${MANIFEST_FILE:-}" ]]; then
      echo "Set MANIFEST_FILE to the dataset manifest to promote, or PROMOTE_MANIFEST=0." >&2
      exit 2
    fi
    ;;
  *)
    echo "DATASET_PROFILE must be 'mini' or 'full', got: $DATASET_PROFILE" >&2
    exit 2
    ;;
esac

colab::section "Download plan"
PLAN="$DRIVE_DATASETS/${DATASET_PROFILE}_download_plan.json"
python -m src.data.build_episode_download_plan \
  --selection "$SELECTION_FILE" \
  --include-videos \
  --output "$PLAN"

colab::section "Download episodes to $SOURCE_ROOT"
python -m src.data.download_selected_episodes \
  --plan "$PLAN" \
  --output-dir "$SOURCE_ROOT"

colab::section "LeRobot -> RLDS, parity, and OpenVLA batch compatibility"
PROJECT_ROOT="$PROJECT_ROOT" \
OPENVLA_ROOT="$OPENVLA_ROOT" \
SOURCE_ROOT="$SOURCE_ROOT" \
SELECTION_FILE="$SELECTION_FILE" \
TFDS_ROOT="$RLDS_ROOT" \
BASE_CHECKPOINT="$BASE_CHECKPOINT" \
ARTIFACT_ROOT="$ARTIFACT_ROOT" \
PARITY_EPISODES_PER_SPLIT="$PARITY_EPISODES_PER_SPLIT" \
COMPATIBILITY_SAMPLES_PER_SPLIT="${COMPATIBILITY_SAMPLES_PER_SPLIT:-4}" \
PROMOTE_MANIFEST="$PROMOTE_MANIFEST" \
MANIFEST_FILE="${MANIFEST_FILE:-}" \
  bash training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh

colab::report_disk
colab::section "Dataset prepare complete"
echo "Profile: $DATASET_PROFILE"
echo "RLDS:    $RLDS_ROOT/$DATASET_NAME/1.0.0"
echo "Reports: $ARTIFACT_ROOT"
echo "03_stage_a_train reads DATA_ROOT_DIR=$RLDS_ROOT"
