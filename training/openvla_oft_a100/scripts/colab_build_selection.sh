#!/usr/bin/env bash
# 800 EpisodeのStage A Selectionをメタデータから生成する。
#
#   bash training/openvla_oft_a100/scripts/colab_build_selection.sh
#
# LeRobotのmeta/だけを取得するので約50MB・数十秒で終わり、GPUも要らない。Episode本体は
# ここでは落とさない（colab_dataset_prepare.shの担当）。
#
# 出力はseedと固定revisionから決まるため再現する。Driveへ置くのは同じ選定を
# 別セッションで作り直さずに済ませるためで、消えても再生成できる。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

# configs/datasets/dataset_sources.yaml の sylvest_libero_plus_lerobot と一致させる。
REPO_ID="${SELECTION_REPO_ID:-Sylvest/libero_plus_lerobot}"
REVISION="${SELECTION_REVISION:-22c57433fef692b5b9ecc0795344daac7fa867a5}"
# configs/datasets/stage_a_balanced_v001.yaml の seed と final_split_per_task。
SEED="${SELECTION_SEED:-20260802}"
PER_TASK="${SELECTION_PER_TASK:-20}"
TRAIN_PER_TASK="${SELECTION_TRAIN_PER_TASK:-16}"
DATASET_ID="${SELECTION_DATASET_ID:-parc_stage_a_balanced_v001}"

META_DIR="$WORK_ROOT/selection/meta"
INVENTORY="$WORK_ROOT/selection/inventory.json"
SELECTION="$DRIVE_DATASETS/libero_plus_selection_v001.json"
MANIFEST="$DRIVE_DATASETS/dataset_manifest.json"

colab::require_drive
mkdir -p "$DRIVE_DATASETS" "$META_DIR"

if [[ -f "$SELECTION" ]] && [[ "${FORCE_SELECTION:-0}" != "1" ]]; then
  echo "Selection already exists: $SELECTION"
  echo "Set FORCE_SELECTION=1 to rebuild it."
  exit 0
fi

colab::section "Metadata only (about 50MB; no episode payloads)"
python -m src.data.download_hf_metadata \
  --repo-id "$REPO_ID" \
  --revision "$REVISION" \
  --output-dir "$META_DIR" \
  --format lerobot_v2_1

colab::section "Inventory"
python -m src.data.inspect_lerobot_metadata \
  --meta-dir "$META_DIR/meta" \
  --suite-map configs/datasets/libero_task_suite_map.json \
  --repo-id "$REPO_ID" \
  --revision "$REVISION" \
  --license mit \
  --output "$INVENTORY"

colab::section "Balanced selection: 40 tasks x $PER_TASK episodes"
python -m src.data.select_balanced_episodes \
  --inventory "$INVENTORY" \
  --per-task "$PER_TASK" \
  --train-per-task "$TRAIN_PER_TASK" \
  --seed "$SEED" \
  --output "$SELECTION"

colab::section "Dataset manifest"
python -m src.data.build_dataset_manifest \
  --inventory "$INVENTORY" \
  --selection "$SELECTION" \
  --dataset-id "$DATASET_ID" \
  --output "$MANIFEST"

python -m src.data.validate_dataset_manifest "$MANIFEST" \
  --schema schemas/dataset_manifest.schema.json

colab::section "Selection ready"
echo "Selection: $SELECTION"
echo "Manifest:  $MANIFEST"
echo "Next: bash training/openvla_oft_a100/scripts/colab_dataset_prepare.sh"
