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
colab::notify_on_exit "Dataset prepare ($DATASET_PROFILE)"

cd "$PROJECT_ROOT"
colab::require_drive

DRIVE_RLDS="$DRIVE_RLDS_ROOT/$DATASET_PROFILE"
RLDS_BUILDER="$RLDS_ROOT/$DATASET_NAME/$DATASET_VERSION"

# Converting 800 episodes costs about an hour and /content does not survive the
# VM. If a finished dataset is already on Drive, copy it back instead of
# downloading and converting again.
#
# Completeness is measured in bytes, not by the presence of dataset_info.json.
# rsync copies in name order, so an interrupted restore leaves that file in place
# with shards still missing, and the next run would skip the restore and train on
# a dataset with holes in it. rsync is incremental, so re-syncing a nearly
# complete tree is cheap.
drive_rlds_bytes=0
work_rlds_bytes=0
if [[ -d "$DRIVE_RLDS" ]]; then
  drive_rlds_bytes=$(colab::tree_bytes "$DRIVE_RLDS")
  [[ -d "$RLDS_ROOT" ]] && work_rlds_bytes=$(colab::tree_bytes "$RLDS_ROOT")
fi

if [[ "${PERSIST_RLDS:-1}" == "1" ]] \
  && (( drive_rlds_bytes > 0 )) \
  && [[ -f "$DRIVE_RLDS/$DATASET_NAME/$DATASET_VERSION/dataset_info.json" ]] \
  && [[ "${FORCE_RECONVERT:-0}" != "1" ]]
then
  if (( work_rlds_bytes == drive_rlds_bytes )); then
    colab::section "RLDS already restored"
    echo "$RLDS_BUILDER matches Drive ($((drive_rlds_bytes / 1024**3))GB)."
    echo "Set FORCE_RECONVERT=1 to rebuild it from source instead."
    colab::section "Dataset prepare complete (already present)"
    echo "RLDS: $RLDS_BUILDER"
    echo "03_stage_a_train reads DATA_ROOT_DIR=$RLDS_ROOT"
    exit 0
  fi

  colab::section "Restoring RLDS from Drive"
  # A restore that runs out of disk leaves a partial dataset whose
  # dataset_info.json is present, so the next run passes its check and trains on
  # missing shards without complaining. Refuse before writing anything. Only the
  # shortfall has to fit: rsync leaves what already matches alone.
  restore_bytes=$(( drive_rlds_bytes - work_rlds_bytes ))
  restore_free=$(colab::free_bytes "$WORK_ROOT")
  if (( restore_bytes + 1073741824 > restore_free )); then
    echo "Not enough space to restore the RLDS: needs about" >&2
    echo "  $((restore_bytes / 1024**3))GB plus margin, $((restore_free / 1024**3))GB free at $WORK_ROOT." >&2
    echo "The submission archive under $SUBMISSION_BUILD_ROOT is the usual thing" >&2
    echo "to remove, once it has been downloaded." >&2
    exit 1
  fi
  colab::sync_tree "$DRIVE_RLDS" "$RLDS_ROOT"

  # rsync exiting 0 is not the same as the tree matching: a source read error or
  # a full disk can end it cleanly with files missing.
  restored_bytes=$(colab::tree_bytes "$RLDS_ROOT")
  if (( restored_bytes != drive_rlds_bytes )); then
    echo "Restore is short: $restored_bytes of $drive_rlds_bytes bytes." >&2
    echo "Re-run this script; rsync will copy only what is missing." >&2
    exit 1
  fi
  echo "Restored: $RLDS_BUILDER"
  echo "Set FORCE_RECONVERT=1 to rebuild it from source instead."
  colab::report_disk
  colab::section "Dataset prepare complete (restored)"
  echo "RLDS: $RLDS_BUILDER"
  echo "03_stage_a_train reads DATA_ROOT_DIR=$RLDS_ROOT"
  exit 0
fi


FULL_SELECTION="${FULL_SELECTION:-$DRIVE_DATASETS/libero_plus_selection_v001.json}"
if [[ ! -f "$FULL_SELECTION" ]]; then
  # Deterministic from the pinned revision and seed, and metadata-only, so
  # building it is cheaper than making the caller go and find it.
  echo "Selection not found; building it from metadata: $FULL_SELECTION"
  bash "$SCRIPT_DIR/colab_build_selection.sh"
fi
if [[ ! -f "$FULL_SELECTION" ]]; then
  echo "Selection file still not found: $FULL_SELECTION" >&2
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

# validate_rlds_batch_transform.py reads AutoProcessor and AutoConfig only, never
# the weights, so dataset preparation does not need the 15GB checkpoint. On a
# runtime that has not run colab_setup.sh -- a free T4, say, since none of this
# path touches the GPU -- fetch just the processor files instead.
if [[ ! -d "$BASE_CHECKPOINT" ]] || [[ "${PROCESSOR_ONLY_CHECKPOINT:-0}" == "1" ]]; then
  BASE_CHECKPOINT="$WORK_ROOT/models/openvla_oft_plus_processor"
  if [[ ! -f "$BASE_CHECKPOINT/config.json" ]]; then
    colab::section "Processor and config only (no weights)"
    mkdir -p "$BASE_CHECKPOINT"
    CHECKPOINT_REPO="${CHECKPOINT_REPO:-Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata}" \
    PROCESSOR_DIR="$BASE_CHECKPOINT" python - <<'PY'
import os

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["CHECKPOINT_REPO"],
    local_dir=os.environ["PROCESSOR_DIR"],
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer*",
        "dataset_statistics.json",
        # trust_remote_code=True loads the custom prismatic classes from here.
        "*.py",
    ],
)
print("Processor files at:", path)
PY
  fi
  echo "Using processor-only checkpoint: $BASE_CHECKPOINT"
fi

# The full profile fetches about 2,400 files. Anonymous access does not finish
# that, and huggingface_hub retries 429 internally, so the failure looks like the
# download going quiet for minutes rather than an error. Settle the token here,
# after the cheap argument checks and before anything long starts.
if [[ "$DATASET_PROFILE" == "full" ]]; then
  colab::section "Hugging Face token"
  python -m src.data.hf_download
fi

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

if [[ "${PERSIST_RLDS:-1}" == "1" ]]; then
  colab::section "Persisting RLDS to Drive"
  # Failing here must not discard an hour of conversion, so do not let the
  # free-space refusal take the script down.
  colab::persist_dataset "$RLDS_ROOT" "$DRIVE_RLDS" || true
fi

colab::report_disk
colab::section "Dataset prepare complete"
echo "Profile: $DATASET_PROFILE"
echo "RLDS:    $RLDS_ROOT/$DATASET_NAME/1.0.0"
echo "Reports: $ARTIFACT_ROOT"
echo "03_stage_a_train reads DATA_ROOT_DIR=$RLDS_ROOT"
