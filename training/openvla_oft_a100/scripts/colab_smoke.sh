#!/usr/bin/env bash
# Free-tier T4 smoke: exercise the whole pipeline plumbing without an A100, the
# 15GB checkpoint, or Google Drive.
#
#   bash training/openvla_oft_a100/scripts/colab_smoke.sh
#
# What this covers:
#   - bootstrap_colab.sh, including the Python 3.12 TensorFlow dependency patch
#   - the vendored prismatic runtime for the submission package
#   - LeRobot -> RLDS conversion, source parity, and the PARC dataset registry
#   - RLDSBatchTransform, which loads only AutoProcessor/AutoConfig, not weights
#
# What this cannot cover: Stage A training, the GPU gate in notebook 06, and the
# submission ZIP all need the real 7B checkpoint on an A100. There is no small
# drop-in replacement for OpenVLA-OFT, so those stay A100-only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

SMOKE_ROOT="${SMOKE_ROOT:-$WORK_ROOT/smoke}"
SMOKE_SOURCE="$SMOKE_ROOT/source"
SMOKE_RLDS="$SMOKE_ROOT/rlds"
SMOKE_SELECTION="$SMOKE_ROOT/selection.json"
SMOKE_ARTIFACTS="$SMOKE_ROOT/reports"
# Processor and config files only. AutoProcessor and AutoConfig are all the batch
# transform needs, so the multi-gigabyte safetensors shards stay unfetched.
SMOKE_PROCESSOR="$SMOKE_ROOT/processor"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata}"

colab::section "Runtime"
python scripts/check_colab_runtime.py --require-cuda
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

colab::section "OpenVLA-OFT environment"
PROJECT_ROOT="$PROJECT_ROOT" WORKDIR="$OPENVLA_ROOT" \
  bash training/openvla_oft_a100/scripts/bootstrap_colab.sh

colab::section "Submission runtime vendor"
OPENVLA_OFT_SOURCE="$OPENVLA_ROOT" \
  bash submission/openvla_oft_offline/scripts/prepare_vendor.sh

colab::section "Data dependencies"
python -m pip install -q -r training/openvla_oft_a100/requirements-data.txt

colab::section "Processor and config only (no weights)"
mkdir -p "$SMOKE_PROCESSOR"
CHECKPOINT_REPO="$CHECKPOINT_REPO" SMOKE_PROCESSOR="$SMOKE_PROCESSOR" python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["CHECKPOINT_REPO"],
    local_dir=os.environ["SMOKE_PROCESSOR"],
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer*",
        "dataset_statistics.json",
        # trust_remote_code=True loads the custom prismatic classes from the repo.
        "*.py",
    ],
)
print("Processor files at:", path)
PY

colab::section "Synthetic LeRobot fixture"
rm -rf "$SMOKE_SOURCE" "$SMOKE_RLDS"
mkdir -p "$SMOKE_SOURCE" "$SMOKE_RLDS" "$SMOKE_ARTIFACTS"
python -m src.data.generate_synthetic_lerobot_fixture \
  --root "$SMOKE_SOURCE" \
  --selection "$SMOKE_SELECTION" \
  --frame-count "${SMOKE_FRAME_COUNT:-16}" \
  --image-size "${SMOKE_IMAGE_SIZE:-256}"

colab::section "RLDS conversion, parity, and batch compatibility"
# The same script notebook 02 runs, so a pass here exercises the real path.
PROJECT_ROOT="$PROJECT_ROOT" \
OPENVLA_ROOT="$OPENVLA_ROOT" \
SOURCE_ROOT="$SMOKE_SOURCE" \
SELECTION_FILE="$SMOKE_SELECTION" \
TFDS_ROOT="$SMOKE_RLDS" \
BASE_CHECKPOINT="$SMOKE_PROCESSOR" \
ARTIFACT_ROOT="$SMOKE_ARTIFACTS" \
PARITY_EPISODES_PER_SPLIT=1 \
COMPATIBILITY_SAMPLES_PER_SPLIT="${SMOKE_SAMPLES:-2}" \
PROMOTE_MANIFEST=0 \
  bash training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh

colab::section "Reports"
SMOKE_ARTIFACTS="$SMOKE_ARTIFACTS" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["SMOKE_ARTIFACTS"])
conversion = json.loads((root / "rlds_conversion_report.json").read_text(encoding="utf-8"))
print("conversion   ->", conversion["episode_counts"], conversion["contract"])

failed = False
for name in ("rlds_source_parity.json", "openvla_rlds_compatibility.json"):
    report = json.loads((root / name).read_text(encoding="utf-8"))
    status = report["status"]
    print(f"{name:35} -> {status}")
    failed |= status != "pass"
sys.exit(1 if failed else 0)
PY

colab::report_disk
colab::section "T4 smoke passed"
echo "Reports:   $SMOKE_ARTIFACTS"
echo "Not covered on T4: Stage A training, notebook 06 GPU gate, submission ZIP."
echo "Those need the real 7B checkpoint on an A100."
