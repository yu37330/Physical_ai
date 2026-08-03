#!/usr/bin/env bash
# Notebook 00 equivalent: gate the runtime before anything expensive runs.
#
#   bash training/openvla_oft_a100/scripts/colab_preflight.sh
#   REQUIRE_A100=0 bash .../colab_preflight.sh   # free-tier T4 smoke
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
preflight_args=(
  --project-root "$PROJECT_ROOT"
  --work-root "$WORK_ROOT"
  --drive-root "$DRIVE_ROOT"
  --minimum-drive-free-gb "${MINIMUM_DRIVE_FREE_GB:-2}"
  --minimum-work-free-gb "${MINIMUM_WORK_FREE_GB:-80}"
  --output "$DRIVE_ADMIN/preflight.json"
)
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
echo "Report: $DRIVE_ADMIN/preflight.json"
