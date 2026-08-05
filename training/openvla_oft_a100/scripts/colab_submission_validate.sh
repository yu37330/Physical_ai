#!/usr/bin/env bash
# Notebook 04 equivalent: build and validate the submission under /content/work.
#
#   bash training/openvla_oft_a100/scripts/colab_submission_validate.sh
#   RUN_DYNAMIC_SMOKE=1 bash .../colab_submission_validate.sh   # adds the GPU smoke
#
# The ZIP carries the full 7B weights (configs/models/openvla_oft_plus_checkpoint.yaml
# budgets it at up to 20GB), so it is built on /content and never copied to Drive.
# Download it to the local machine from the VS Code Colab file view. Drive keeps
# only the build manifest and its SHA256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

SUBMISSION_DIR="$PROJECT_ROOT/submission/openvla_oft_offline"
MODEL_TARGET="$SUBMISSION_DIR/model_weights/openvla_oft_plus"
ZIP_PATH="$SUBMISSION_BUILD_ROOT/parc2026_track1_openvla_oft_plus.zip"

colab::section "Submission dependency policy"
python scripts/validate_submission_requirements.py submission

colab::section "Official validator"
bash scripts/fetch_official_repo.sh "$OFFICIAL_REPO_ROOT"
OFFICIAL_COMMIT="$(git -C "$OFFICIAL_REPO_ROOT" rev-parse HEAD)"
echo "Official commit: $OFFICIAL_COMMIT"

# Assemble from the Stage A run rather than expecting a hand-placed checkpoint.
# Base weights alone would be "公開重みを実質的に変更せず推論する", which the rules
# disallow; the trained action head and proprio projector are what make the
# submission independently trained.
STAGE_A_RUN="${STAGE_A_RUN:-$RUN_ROOT/stage_a_s2_head_proprio_500}"
if [[ ! -d "$STAGE_A_RUN" ]]; then
  STAGE_A_RUN="$RUN_ROOT/stage_a_s1_head_proprio_100"
fi

if [[ "${ASSEMBLE_CHECKPOINT:-1}" == "1" ]]; then
  if [[ ! -d "$STAGE_A_RUN" ]]; then
    echo "No Stage A run found under $RUN_ROOT." >&2
    echo "Run colab_stage_a.sh s1 first, or set STAGE_A_RUN." >&2
    exit 1
  fi
  colab::section "Assembling the submission checkpoint from $(basename "$STAGE_A_RUN")"
  python scripts/assemble_submission_checkpoint.py \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --trained-run-dir "$STAGE_A_RUN" \
    --output "$MODEL_TARGET" \
    --report "$SUBMISSION_BUILD_ROOT/assembled_checkpoint.json"
  # The weights just changed, so any existing archive is of a different model.
  FORCE_ZIP=1
fi

if [[ ! -d "$MODEL_TARGET" ]] || [[ -z "$(ls -A "$MODEL_TARGET" 2>/dev/null)" ]]; then
  mkdir -p "$MODEL_TARGET"
  echo "Place the frozen final checkpoint at: $MODEL_TARGET" >&2
  echo "Then re-run this script, or use ASSEMBLE_CHECKPOINT=1." >&2
  exit 1
fi

# The ZIP builder already skips these, but the directory check reports them and
# they only appear once something has imported the runtime.
find "$SUBMISSION_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

colab::section "Static validation (directory)"
python "$OFFICIAL_REPO_ROOT/validate_submission.py" "$SUBMISSION_DIR" \
  --static --pip-dry-run --json

# Rebuilding needs room for a second copy of a 14GB archive, and the existing one
# is already validated. FORCE_ZIP=1 rebuilds after the checkpoint changes.
if [[ -f "$ZIP_PATH" ]] && [[ "${FORCE_ZIP:-0}" != "1" ]]; then
  colab::section "Reusing the existing ZIP"
  echo "$ZIP_PATH"
  echo "Set FORCE_ZIP=1 to rebuild it, for instance after reassembling the checkpoint."
  [[ -f "$ZIP_PATH.json" ]] && cat "$ZIP_PATH.json"
else
  colab::section "Build ZIP on $SUBMISSION_BUILD_ROOT"
  mkdir -p "$SUBMISSION_BUILD_ROOT"
  python submission/openvla_oft_offline/tools/build_submission_zip.py \
    --source "$SUBMISSION_DIR" \
    --output "$ZIP_PATH"
  cat "$ZIP_PATH.json"
fi

colab::section "Static validation (ZIP)"
python "$OFFICIAL_REPO_ROOT/validate_submission.py" "$ZIP_PATH" \
  --static --pip-dry-run --json

if [[ "${RUN_DYNAMIC_SMOKE:-0}" == "1" ]]; then
  # Against the directory, not the ZIP: validating the archive extracts it first,
  # which needs another ~14GB on a disk that already holds both the checkpoint and
  # the archive. The contents are identical and the archive's own integrity was
  # just checked statically. SMOKE_TARGET forces the ZIP where space allows.
  SMOKE_TARGET="${SMOKE_TARGET:-$SUBMISSION_DIR}"
  colab::section "Dynamic smoke (/health, /reset, /act) against $(basename "$SMOKE_TARGET")"
  python "$OFFICIAL_REPO_ROOT/validate_submission.py" "$SMOKE_TARGET" \
    --camera 128 --health-timeout "${SMOKE_HEALTH_TIMEOUT:-180}" --json
else
  echo
  echo "Skipped the dynamic smoke. Re-run with RUN_DYNAMIC_SMOKE=1 once static passes."
fi

colab::section "Persist build record to Drive"
if colab::require_drive; then
  colab::persist "$ZIP_PATH.json" \
    "$DRIVE_SUBMISSIONS/parc2026_track1_openvla_oft_plus.zip.json"
  {
    echo "official_commit: $OFFICIAL_COMMIT"
    echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "$ZIP_PATH"
  } > "$SUBMISSION_BUILD_ROOT/submission_sha256.txt"
  colab::persist "$SUBMISSION_BUILD_ROOT/submission_sha256.txt" \
    "$DRIVE_SUBMISSIONS/submission_sha256.txt"
fi

colab::report_disk
colab::section "Submission build complete"
echo "ZIP: $ZIP_PATH"
echo "Download it to the local machine; it is too large for Drive and /content is ephemeral."
