#!/usr/bin/env bash
# Notebook 07 equivalent: fix requirements.txt inside an already-built archive.
#
#   bash training/openvla_oft_a100/scripts/colab_patch_submission.sh
#   SOURCE_ZIP=/content/work/submission/x.zip bash .../colab_patch_submission.sh
#
# The scored run failed at `import torch` on a missing libnvJitLink.so.12, which
# is a requirements.txt problem alone -- the weights, the assembly and the
# validator results all still stand. Rebuilding through
# colab_submission_validate.sh would re-download the 15GB base checkpoint and
# reassemble it for a change to one text file, so this reads the archive that is
# already on Drive and rewrites that single entry.
#
# The source archive is never modified. If anything here fails, the original is
# still on Drive and still submittable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"
colab::notify_on_exit "Submission requirements patch"

cd "$PROJECT_ROOT"

ARCHIVE_NAME="${ARCHIVE_NAME:-parc2026_track1_openvla_oft_plus.zip}"
SOURCE_ZIP="${SOURCE_ZIP:-$DRIVE_SUBMISSIONS/$ARCHIVE_NAME}"
# Written to /content, not Drive: the patched archive is built by streaming the
# source through, and doing that with both ends on the FUSE mount is both slow
# and the easiest way to end up with a truncated 14GB file.
PATCHED_NAME="${PATCHED_NAME:-${ARCHIVE_NAME%.zip}_patched.zip}"
OUTPUT_ZIP="${OUTPUT_ZIP:-$SUBMISSION_BUILD_ROOT/$PATCHED_NAME}"
REQUIREMENTS="${REQUIREMENTS:-$PROJECT_ROOT/submission/openvla_oft_offline/requirements.txt}"

colab::require_drive

if [[ ! -f "$SOURCE_ZIP" ]]; then
  echo "Submission archive not found: $SOURCE_ZIP" >&2
  echo "Set SOURCE_ZIP to the archive to patch. What Drive currently holds:" >&2
  ls -la "$DRIVE_SUBMISSIONS" 2>/dev/null >&2 || echo "  (no $DRIVE_SUBMISSIONS)" >&2
  exit 1
fi

colab::section "Submission dependency policy"
python scripts/validate_submission_requirements.py submission

# Confirm the checked-out tree actually carries the fix before spending twenty
# minutes copying 14GB. Patching an archive with the same broken file is the one
# outcome this script must not produce quietly.
colab::section "Replacement requirements.txt"
if ! grep -q '^nvidia-nvjitlink-cu12==' "$REQUIREMENTS"; then
  echo "$REQUIREMENTS has no pinned nvidia-nvjitlink-cu12." >&2
  echo "That pin is the fix; without it the patched archive fails the same way." >&2
  echo "Check out a commit that has it, or set REQUIREMENTS." >&2
  exit 1
fi
cat "$REQUIREMENTS"

colab::section "Patching $ARCHIVE_NAME"
mkdir -p "$SUBMISSION_BUILD_ROOT"
python scripts/patch_submission_zip.py \
  --input "$SOURCE_ZIP" \
  --output "$OUTPUT_ZIP" \
  --requirements "$REQUIREMENTS"

colab::section "Official validator"
bash scripts/fetch_official_repo.sh "$OFFICIAL_REPO_ROOT"
OFFICIAL_COMMIT="$(git -C "$OFFICIAL_REPO_ROOT" rev-parse HEAD)"
echo "Official commit: $OFFICIAL_COMMIT"

# Against the archive rather than a directory: the archive is what changed, and
# --pip-dry-run is the check that matters here. It resolves the requirements the
# way the scoring image will, so a pin that cannot be satisfied surfaces now
# instead of after a 14GB upload.
colab::section "Static validation (ZIP)"
python "$OFFICIAL_REPO_ROOT/validate_submission.py" "$OUTPUT_ZIP" \
  --static --pip-dry-run --json

colab::section "Copy the patched archive to Drive"
zip_bytes=$(stat -c %s "$OUTPUT_ZIP")
drive_copy="$DRIVE_SUBMISSIONS/$PATCHED_NAME"
drive_free=$(colab::free_bytes "$DRIVE_ROOT")
existing_bytes=0
[[ -f "$drive_copy" ]] && existing_bytes=$(stat -c %s "$drive_copy")

if (( zip_bytes + 1073741824 > drive_free + existing_bytes )); then
  echo "Not enough room on Drive: needs $((zip_bytes / 1024**3))GB plus margin," >&2
  echo "  $((drive_free / 1024**3))GB free. The patched archive is at $OUTPUT_ZIP." >&2
  echo "  Free space on Drive, or download it from /content directly." >&2
  exit 1
fi

cp "$OUTPUT_ZIP" "$drive_copy"
# A FUSE write that runs short leaves a plausible-looking file, and the next
# thing that touches it is a 14GB download over a home connection.
copied_bytes=$(stat -c %s "$drive_copy")
if (( copied_bytes != zip_bytes )); then
  rm -f "$drive_copy"
  echo "Copy to Drive was short ($copied_bytes of $zip_bytes bytes); removed it." >&2
  exit 1
fi
echo "$drive_copy"

colab::persist "$OUTPUT_ZIP.json" "$DRIVE_SUBMISSIONS/$PATCHED_NAME.json"
{
  echo "official_commit: $OFFICIAL_COMMIT"
  echo "patched_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_archive: $SOURCE_ZIP"
  sha256sum "$OUTPUT_ZIP"
} > "$SUBMISSION_BUILD_ROOT/patched_submission_sha256.txt"
colab::persist "$SUBMISSION_BUILD_ROOT/patched_submission_sha256.txt" \
  "$DRIVE_SUBMISSIONS/patched_submission_sha256.txt"

colab::report_disk
colab::section "Patch complete"
echo "Submit:   $drive_copy"
echo "SHA256:   $DRIVE_SUBMISSIONS/patched_submission_sha256.txt"
echo "Original: $SOURCE_ZIP (unchanged)"
echo
echo "Download the patched archive and check its SHA256 before submitting."
