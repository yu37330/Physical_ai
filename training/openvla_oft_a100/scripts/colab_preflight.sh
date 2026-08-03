#!/usr/bin/env bash
# Notebook 00 equivalent: gate the runtime before anything expensive runs.
#
#   bash training/openvla_oft_a100/scripts/colab_preflight.sh
#
# Assumes Google Drive is already mounted. Mount it from a notebook cell or with
# the 'Colab: Mount Google Drive to Server...' command in VS Code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

colab::section "Runtime"
python scripts/check_colab_runtime.py --require-cuda

colab::section "Preflight"
colab::require_drive
mkdir -p "$DRIVE_ADMIN" "$WORK_ROOT"
# Drive now holds only manifests, reports and trained components, so demanding
# tens of gigabytes free would fail the gate for no reason. The heavy artifacts
# live under $WORK_ROOT, which is checked against --minimum-work-free-gb.
python training/openvla_oft_a100/scripts/preflight.py \
  --project-root "$PROJECT_ROOT" \
  --work-root "$WORK_ROOT" \
  --drive-root "$DRIVE_ROOT" \
  --minimum-drive-free-gb "${MINIMUM_DRIVE_FREE_GB:-2}" \
  --minimum-work-free-gb "${MINIMUM_WORK_FREE_GB:-80}" \
  --require-a100-40gb \
  --output "$DRIVE_ADMIN/preflight.json"

colab::report_disk
colab::section "Preflight passed"
echo "Report: $DRIVE_ADMIN/preflight.json"
