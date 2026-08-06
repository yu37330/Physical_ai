#!/usr/bin/env bash
# Carry the submission archive off Colab through Google Drive, one part at a time.
#
#   bash training/openvla_oft_a100/scripts/colab_transfer_submission.sh next
#   bash training/openvla_oft_a100/scripts/colab_transfer_submission.sh done
#   bash training/openvla_oft_a100/scripts/colab_transfer_submission.sh status
#
# VS Code downloads through the Jupyter contents API, which cannot serve a file
# this size. Drive can, but only has about 1.5GB free, so a part is staged there,
# downloaded from drive.google.com, and deleted before the next one is carved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

COMMAND="${1:-status}"
case "$COMMAND" in
  next|done|status) ;;
  *) echo "Usage: $0 {next|done|status}" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
colab::require_drive

SOURCE_ZIP="${SOURCE_ZIP:-$PROJECT_ROOT/parc2026_track1_openvla_oft_plus.zip}"
if [[ ! -f "$SOURCE_ZIP" ]]; then
  # colab_submission_validate.sh leaves it here; it only moves if someone moved it.
  SOURCE_ZIP="$SUBMISSION_BUILD_ROOT/parc2026_track1_openvla_oft_plus.zip"
fi
if [[ ! -f "$SOURCE_ZIP" ]]; then
  echo "Submission archive not found. Set SOURCE_ZIP to its path." >&2
  exit 1
fi

STAGING="${TRANSFER_STAGING:-$DRIVE_SUBMISSIONS/transfer}"

python scripts/transfer_large_file.py "$COMMAND" \
  --source "$SOURCE_ZIP" \
  --staging "$STAGING" \
  --part-mib "${TRANSFER_PART_MIB:-1024}"

if [[ "$COMMAND" == "next" ]]; then
  echo
  echo "Open https://drive.google.com/ and download it from:"
  echo "  PARC2026 / 60_submissions / transfer"
  echo "Then run: bash $0 done"
fi
if [[ "$COMMAND" == "done" ]]; then
  echo
  # Drive counts trashed files against the quota, so a delete that looks like it
  # freed 1GB can leave the next part with nowhere to go.
  df -h "$DRIVE_ROOT" | tail -1
  echo "If the free space did not move, empty the Drive trash (ゴミ箱) and re-check."
fi
