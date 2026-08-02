#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OPENVLA_ROOT="${OPENVLA_ROOT:-/content/openvla-oft}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the selected LeRobot payload root}"
SELECTION_FILE="${SELECTION_FILE:?Set SELECTION_FILE to the selected episode JSON}"
TFDS_ROOT="${TFDS_ROOT:?Set TFDS_ROOT to the output TFDS/RLDS root}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the local OpenVLA-OFT+ checkpoint}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/datasets/parc_stage_a_balanced_v001}"
PARITY_EPISODES_PER_SPLIT="${PARITY_EPISODES_PER_SPLIT:-2}"
COMPATIBILITY_SAMPLES_PER_SPLIT="${COMPATIBILITY_SAMPLES_PER_SPLIT:-4}"
PROMOTE_MANIFEST="${PROMOTE_MANIFEST:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-}"

if [[ "$PROMOTE_MANIFEST" == "1" && -z "$MANIFEST_FILE" ]]; then
  echo "MANIFEST_FILE is required when PROMOTE_MANIFEST=1" >&2
  exit 2
fi

export PYTHONPATH="$OPENVLA_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$ARTIFACT_ROOT" "$TFDS_ROOT"
python -m pip install -r "$PROJECT_ROOT/training/openvla_oft_a100/requirements-data.txt"

python -m src.data.convert_selected_lerobot_to_rlds \
  --source-root "$SOURCE_ROOT" \
  --selection "$SELECTION_FILE" \
  --output-root "$TFDS_ROOT" \
  --report "$ARTIFACT_ROOT/rlds_conversion_report.json"

python -m src.data.validate_rlds_source_parity \
  --source-root "$SOURCE_ROOT" \
  --tfds-root "$TFDS_ROOT" \
  --episodes-per-split "$PARITY_EPISODES_PER_SPLIT" \
  --output "$ARTIFACT_ROOT/rlds_source_parity.json"

python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_parc_dataset_registry.py" \
  --openvla-root "$OPENVLA_ROOT"

python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/validate_rlds_batch_transform.py" \
  --openvla-root "$OPENVLA_ROOT" \
  --data-root "$TFDS_ROOT" \
  --checkpoint "$BASE_CHECKPOINT" \
  --dataset-name parc_libero_plus_selected \
  --samples-per-split "$COMPATIBILITY_SAMPLES_PER_SPLIT" \
  --output "$ARTIFACT_ROOT/openvla_rlds_compatibility.json"

if [[ "$PROMOTE_MANIFEST" == "1" ]]; then
  python -m src.data.update_dataset_manifest_after_rlds \
    --manifest "$MANIFEST_FILE" \
    --conversion-report "$ARTIFACT_ROOT/rlds_conversion_report.json" \
    --parity-report "$ARTIFACT_ROOT/rlds_source_parity.json" \
    --compatibility-report "$ARTIFACT_ROOT/openvla_rlds_compatibility.json" \
    --output "$ARTIFACT_ROOT/dataset_manifest.payload_validated.json"

  python -m src.data.validate_dataset_manifest \
    "$ARTIFACT_ROOT/dataset_manifest.payload_validated.json" \
    --schema "$PROJECT_ROOT/schemas/dataset_manifest.schema.json"
else
  echo "Mini/smoke mode: skipped production Dataset Manifest promotion."
fi

echo "Stage A RLDS dataset, source parity, and OpenVLA compatibility checks completed."
echo "Artifacts: $ARTIFACT_ROOT"
