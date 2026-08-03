#!/usr/bin/env bash
# Notebook 00 + 01 equivalent: gate the runtime, build the pinned OpenVLA-OFT
# environment, and fetch the base checkpoint into /content/work.
#
#   bash training/openvla_oft_a100/scripts/colab_setup.sh
#
# Set SKIP_PREFLIGHT=1 to skip the A100/disk gate (e.g. a non-A100 debug run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

# Notebook 00 runs the same script, so set SKIP_PREFLIGHT=1 when the gate has
# already passed in this session.
#
# This stage installs the OpenVLA-OFT environment and the ~15GB checkpoint; the
# 80GB default belongs to the 800 episode RLDS conversion further down the
# pipeline, and demanding it here fails runtimes that could finish setup fine.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  MINIMUM_WORK_FREE_GB="${MINIMUM_WORK_FREE_GB:-40}" bash "$SCRIPT_DIR/colab_preflight.sh"
else
  colab::section "Runtime"
  python scripts/check_colab_runtime.py --require-cuda
fi

colab::section "OpenVLA-OFT environment"
PROJECT_ROOT="$PROJECT_ROOT" WORKDIR="$OPENVLA_ROOT" \
  bash training/openvla_oft_a100/scripts/bootstrap_colab.sh

colab::section "Submission runtime vendor"
OPENVLA_OFT_SOURCE="$OPENVLA_ROOT" \
  bash submission/openvla_oft_offline/scripts/prepare_vendor.sh

colab::section "Data dependencies"
python -m pip install -q -r training/openvla_oft_a100/requirements-data.txt

colab::section "Base checkpoint"
mkdir -p "$BASE_CHECKPOINT"
# Writes checkpoint_manifest.json and model_source_manifest.json alongside the
# weights, and exits non-zero if the manifest does not pass.
python training/openvla_oft_a100/scripts/download_base_checkpoint.py \
  --output "$BASE_CHECKPOINT"

colab::section "Persist small records to Drive"
if colab::require_drive; then
  colab::persist "$BASE_CHECKPOINT/checkpoint_manifest.json" \
    "$DRIVE_MODELS/openvla_oft_plus_base/checkpoint_manifest.json"
  colab::persist "$BASE_CHECKPOINT/model_source_manifest.json" \
    "$DRIVE_MODELS/openvla_oft_plus_base/model_source_manifest.json"
  mkdir -p "$DRIVE_ADMIN"
  python training/openvla_oft_a100/scripts/capture_environment.py \
    > "$DRIVE_ADMIN/environment.json"
  echo "Persisted $DRIVE_ADMIN/environment.json"
fi

colab::report_disk
colab::section "Setup complete"
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "OpenVLA-OFT:     $OPENVLA_ROOT"
