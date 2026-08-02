#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OPENVLA_ROOT="${OPENVLA_ROOT:-/content/openvla-oft}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the selected LeRobot payload root}"
SELECTION_FILE="${SELECTION_FILE:?Set SELECTION_FILE to libero_plus_selection_v001.json}"
TFDS_ROOT="${TFDS_ROOT:?Set TFDS_ROOT to the output TFDS/RLDS root}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the local OpenVLA-OFT+ checkpoint}"
MANIFEST_FILE="${MANIFEST_FILE:?Set MANIFEST_FILE to the metadata-selected dataset manifest}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/datasets/parc_stage_a_balanced_v001}"

mkdir -p "$ARTIFACT_ROOT" "$TFDS_ROOT"
python -m pip install -r "$PROJECT_ROOT/training/openvla_oft_a100/requirements-data.txt"

python "$PROJECT_ROOT/src/data/convert_selected_lerobot_to_rlds.py" \
  --source-root "$SOURCE_ROOT" \
  --selection "$SELECTION_FILE" \
  --output-root "$TFDS_ROOT" \
  --report "$ARTIFACT_ROOT/rlds_conversion_report.json"

python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_parc_dataset_registry.py" \
  --openvla-root "$OPENVLA_ROOT"

PYTHONPATH="$OPENVLA_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}" \
python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/validate_rlds_batch_transform.py" \
  --openvla-root "$OPENVLA_ROOT" \
  --data-root "$TFDS_ROOT" \
  --checkpoint "$BASE_CHECKPOINT" \
  --dataset-name parc_libero_plus_selected \
  --samples-per-split 4 \
  --output "$ARTIFACT_ROOT/openvla_rlds_compatibility.json"

python "$PROJECT_ROOT/src/data/update_dataset_manifest_after_rlds.py" \
  --manifest "$MANIFEST_FILE" \
  --conversion-report "$ARTIFACT_ROOT/rlds_conversion_report.json" \
  --compatibility-report "$ARTIFACT_ROOT/openvla_rlds_compatibility.json" \
  --output "$ARTIFACT_ROOT/dataset_manifest.payload_validated.json"

python "$PROJECT_ROOT/src/data/validate_dataset_manifest.py" \
  "$ARTIFACT_ROOT/dataset_manifest.payload_validated.json" \
  --schema "$PROJECT_ROOT/schemas/dataset_manifest.schema.json"

echo "Stage A RLDS dataset and OpenVLA compatibility checks completed."
echo "Artifacts: $ARTIFACT_ROOT"
