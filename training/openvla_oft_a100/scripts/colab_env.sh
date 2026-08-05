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
DRIVE_RLDS_ROOT="$DRIVE_ROOT/20_processed/rlds"
DRIVE_MODELS="$DRIVE_ROOT/30_models"
DRIVE_EXPERIMENTS="$DRIVE_ROOT/40_experiments"
DRIVE_DATASETS="$DRIVE_EXPERIMENTS/datasets"
DRIVE_SUBMISSIONS="$DRIVE_ROOT/60_submissions"

# A stalled Hugging Face transfer otherwise hangs forever with the process alive
# and nothing raised to retry on. huggingface_hub reads this at import time.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

# Must match src/data/rlds_contract.py; tests/test_colab_wrappers.py pins that.
DATASET_NAME="${DATASET_NAME:-parc_libero_plus_selected}"
DATASET_VERSION="${DATASET_VERSION:-1.0.0}"
DATASET_MIXTURE="${DATASET_MIXTURE:-parc_stage_a_plus_only}"

colab::section() {
  printf '\n=== %s ===\n' "$1"
}

# Stage A and the submission build run long enough to walk away from, so they
# report their own outcome. Never fatal: a notification that cannot be delivered
# is not a reason to fail a run that just succeeded.
colab::notify() {
  python "$PROJECT_ROOT/scripts/notify_discord.py" \
    --message "$1" --elapsed-seconds "${2:-0}" 2>&1 || true
}

colab::_notify_exit() {
  local code=$?
  if (( code == 0 )); then
    colab::notify "OK: $COLAB_NOTIFY_LABEL" "$SECONDS"
  else
    colab::notify "FAILED (exit $code): $COLAB_NOTIFY_LABEL" "$SECONDS"
  fi
  return "$code"
}

# Call once, near the top of a wrapper. Reports on every exit path, including
# the `set -e` ones, which are the failures worth hearing about.
colab::notify_on_exit() {
  COLAB_NOTIFY_LABEL="$1"
  trap colab::_notify_exit EXIT
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

# Total size of the regular files under a tree, ignoring the directories
# themselves. `du -sb` includes directory inodes, which ext4 reports as 4096 and
# the Drive FUSE mount reports as 0, so comparing two identical trees across the
# two filesystems differs by 4096 per directory and a good copy looks short.
colab::tree_bytes() {
  find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{ total += $1 } END { print total + 0 }'
}

colab::free_bytes() {
  df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '
}

# rsync is present on Colab and preferable for large trees, but not everywhere
# these scripts get exercised.
colab::sync_tree() {
  local source="$1" destination="$2"
  mkdir -p "$destination"
  if command -v rsync > /dev/null; then
    rsync -a --delete "$source/" "$destination/"
  else
    rm -rf "${destination:?}"/*
    cp -a "$source/." "$destination/"
  fi
}

# Converting 800 episodes takes about an hour, and /content is wiped whenever the
# VM goes. The RLDS is roughly 9GB, far past colab::persist's small-artifact
# limit, so it gets its own copy with an explicit free-space check rather than a
# blanket refusal.
colab::persist_dataset() {
  local source="$1" destination="$2"
  if [[ ! -d "$source" ]]; then
    echo "Nothing to persist, missing: $source" >&2
    return 1
  fi
  local needed available existing
  needed=$(colab::tree_bytes "$source")

  # Already there: the space it occupies is not space it needs, so checking free
  # space would refuse a copy that has nothing left to do.
  if [[ -d "$destination" ]]; then
    existing=$(colab::tree_bytes "$destination")
    if (( existing == needed )); then
      echo "Already on Drive, unchanged: $destination"
      return 0
    fi
  fi

  available=$(colab::free_bytes "$DRIVE_ROOT")
  # Keep half a gigabyte spare for manifests and the trained components, which
  # run about 370MB per stage.
  if (( needed + 536870912 > available + ${existing:-0} )); then
    echo "Not copying the dataset to Drive: needs $((needed / 1024**3))GB," >&2
    echo "  only $((available / 1024**3))GB free. Set PERSIST_RLDS=0 to silence this." >&2
    return 1
  fi
  colab::sync_tree "$source" "$destination"
  echo "Persisted $((needed / 1024**2))MB -> $destination"
}

colab::report_disk() {
  colab::section "Disk"
  df -h "$WORK_ROOT" "$DRIVE_ROOT" 2>/dev/null || df -h "$(dirname "$WORK_ROOT")"
}
