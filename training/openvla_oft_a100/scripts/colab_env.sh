#!/usr/bin/env bash
# Shared path contract for Colab runs. Source this; do not execute it.
#
# Storage split (運用原則4):
#   /content/work  ephemeral, ~100GB, fast    -> datasets, base checkpoint, training intermediates
#   Google Drive   persistent, often <5GB     -> manifests, reports, trained components, SHA256
#
# Every wrapper reads the same variables, so a Notebook cell and a Colab
# Terminal invocation cannot drift apart.

PROJECT_ROOT="${PROJECT_ROOT:-/content/Physical_ai}"
WORK_ROOT="${WORK_ROOT:-/content/work}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/PARC2026}"
OPENVLA_ROOT="${OPENVLA_ROOT:-/content/openvla-oft}"

# mini = 3 episode smoke, full = the 800 episode Stage A set.
DATASET_PROFILE="${DATASET_PROFILE:-mini}"

MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$MODEL_ROOT/openvla_oft_plus_base}"
SOURCE_ROOT="${SOURCE_ROOT:-$WORK_ROOT/source/$DATASET_PROFILE}"
# prepare_stage_a_rlds.sh writes here and train_smoke.sh reads from here, so both
# stages stay on one path. Splitting them is what left 03 pointing at a directory
# that 02 never created.
RLDS_ROOT="${RLDS_ROOT:-$WORK_ROOT/rlds/$DATASET_PROFILE}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/runs}"
SUBMISSION_BUILD_ROOT="${SUBMISSION_BUILD_ROOT:-$WORK_ROOT/submission}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/content/PARC2026_pre}"

DRIVE_ADMIN="$DRIVE_ROOT/00_admin"
DRIVE_MODELS="$DRIVE_ROOT/30_models"
DRIVE_EXPERIMENTS="$DRIVE_ROOT/40_experiments"
DRIVE_DATASETS="$DRIVE_EXPERIMENTS/datasets"
DRIVE_SUBMISSIONS="$DRIVE_ROOT/60_submissions"

DATASET_NAME="${DATASET_NAME:-parc_libero_plus_selected}"
DATASET_MIXTURE="${DATASET_MIXTURE:-parc_stage_a_plus_only}"

colab::section() {
  printf '\n=== %s ===\n' "$1"
}

colab::require_drive() {
  if [[ ! -d "$DRIVE_ROOT" ]]; then
    echo "Google Drive is not mounted at $DRIVE_ROOT." >&2
    echo "Run drive.mount('/content/drive') in a notebook cell first," >&2
    echo "or use the 'Colab: Mount Google Drive to Server...' command in VS Code." >&2
    return 1
  fi
}

# Copy a small durable artifact to Drive. Refuses anything large so a stray
# checkpoint cannot silently fill a nearly full Drive.
colab::persist() {
  local source="$1" destination="$2" limit_mb="${3:-256}"
  if [[ ! -e "$source" ]]; then
    echo "Nothing to persist, missing: $source" >&2
    return 1
  fi
  local size_mb
  size_mb=$(du -sm "$source" | cut -f1)
  if (( size_mb > limit_mb )); then
    echo "Refusing to copy ${size_mb}MB to Drive (limit ${limit_mb}MB): $source" >&2
    echo "Keep large artifacts under $WORK_ROOT and download them directly." >&2
    return 1
  fi
  mkdir -p "$(dirname "$destination")"
  cp -R "$source" "$destination"
  echo "Persisted ${size_mb}MB -> $destination"
}

colab::report_disk() {
  colab::section "Disk"
  df -h "$WORK_ROOT" "$DRIVE_ROOT" 2>/dev/null || df -h "$(dirname "$WORK_ROOT")"
}
