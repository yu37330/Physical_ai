#!/usr/bin/env bash
# Notebook 00 equivalent: gate the runtime before anything expensive runs.
#
#   bash training/openvla_oft_a100/scripts/colab_preflight.sh
#   REQUIRE_A100=0 bash .../colab_preflight.sh    # free-tier T4 smoke
#   REQUIRE_DRIVE=0 bash .../colab_preflight.sh   # measurement run, no Drive needed
#
# Mount Google Drive from a notebook cell or with the
# 'Colab: Mount Google Drive to Server...' command in VS Code. Runs that only
# measure something and persist nothing can set REQUIRE_DRIVE=0 instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

colab::section "Runtime"
python scripts/check_colab_runtime.py --require-cuda

colab::section "Preflight"
mkdir -p "$WORK_ROOT"

# Stage A persists manifests and trained components, so it needs Drive. A pure
# measurement run does not, and failing the gate for a missing mount would only
# get in the way.
PREFLIGHT_REPORT="$DRIVE_ADMIN/preflight.json"
if [[ "${REQUIRE_DRIVE:-1}" == "1" ]]; then
  colab::require_drive
  mkdir -p "$DRIVE_ADMIN"
else
  echo "REQUIRE_DRIVE=0: skipping the Drive checks and writing the report locally."
  PREFLIGHT_REPORT="$WORK_ROOT/preflight.json"
fi
# Drive now holds only manifests, reports and trained components, so demanding
# tens of gigabytes free would fail the gate for no reason. The heavy artifacts
# live under $WORK_ROOT, which is checked against --minimum-work-free-gb.
preflight_args=(
  --project-root "$PROJECT_ROOT"
  --work-root "$WORK_ROOT"
  --drive-root "$DRIVE_ROOT"
  --minimum-drive-free-gb "${MINIMUM_DRIVE_FREE_GB:-2}"
  --minimum-work-free-gb "${MINIMUM_WORK_FREE_GB:-80}"
  --output "$PREFLIGHT_REPORT"
)
if [[ "${REQUIRE_DRIVE:-1}" != "1" ]]; then
  # drive_mounted and drive_disk_free stay in the report, just not required.
  preflight_args+=(--no-require-drive)
fi
# Stage A needs an A100 40GB. The T4 smoke only exercises plumbing, so it opts
# out; the a100_40gb check is still reported, just not required.
if [[ "${REQUIRE_A100:-1}" == "1" ]]; then
  preflight_args+=(--require-a100-40gb)
else
  echo "REQUIRE_A100=0: reporting the A100 check without enforcing it."
fi
python training/openvla_oft_a100/scripts/preflight.py "${preflight_args[@]}"

colab::report_disk
colab::section "Preflight passed"
echo "Report: $PREFLIGHT_REPORT"
