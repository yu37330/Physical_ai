#!/usr/bin/env bash
set -euo pipefail

OPENVLA_OFT_COMMIT="${OPENVLA_OFT_COMMIT:-e4287e94541f459edc4feabc4e181f537cd569a8}"
WORKDIR="${WORKDIR:-/content/openvla-oft}"
PROJECT_ROOT="${PROJECT_ROOT:-/content/Physical_ai}"

STAMP="$WORKDIR/.parc_bootstrap_complete"

# Colab sessions drop often. When the VM survives, redoing a 10 minute pip
# install achieves nothing, so skip it -- but only after importing everything and
# checking versions and provenance, never on the strength of the stamp alone.
# colab_action_parity.sh swaps in PyPI transformers, and the check catches that
# too, so training does not silently continue without the fork.
if [[ "${FORCE_BOOTSTRAP:-0}" != "1" ]] \
  && [[ -f "$STAMP" ]] \
  && [[ "$(cat "$STAMP")" == "$OPENVLA_OFT_COMMIT" ]] \
  && python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/check_openvla_env.py" \
       --require-parc-dataset > /dev/null 2>&1
then
  echo "OpenVLA-OFT environment already complete at $OPENVLA_OFT_COMMIT; skipping."
  echo "Set FORCE_BOOTSTRAP=1 to rebuild it."
  exit 0
fi

rm -rf "$WORKDIR"
git clone https://github.com/moojink/openvla-oft.git "$WORKDIR"
git -C "$WORKDIR" checkout "$OPENVLA_OFT_COMMIT"

python -m pip install --upgrade pip setuptools wheel packaging ninja

# Python 3.12 Colab runtimes have no tensorflow==2.15.0 wheel and no
# tensorflow-addons wheel; adjust the pins before the editable install so
# dependency resolution can succeed. No-op on Python 3.10/3.11.
python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_openvla_oft_dependencies.py" \
  "$WORKDIR/pyproject.toml"

python -m pip install -e "$WORKDIR"

# On Python 3.12 the patch drops two dependencies that pip cannot resolve, so
# reinstall them without their own requirement sets. Both only need TensorFlow
# and tensorflow_datasets, which the editable install above already provides.
#
#   tensorflow_graphics -> would pull tensorflow-addons (no 3.12 wheel, no sdist)
#   dlimp               -> pins tensorflow==2.15.0 (no 3.12 wheel)
if ! python -c "import tensorflow_graphics" 2>/dev/null; then
  python -m pip install --no-deps "tensorflow_graphics==2021.12.3"
fi
if ! python -c "import dlimp" 2>/dev/null; then
  python -m pip install --no-deps \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla@${DLIMP_COMMIT:-040105d256bd28866cc6620621a3d5f7b6b91b46}"
fi

# The Colab image ships tensorflow_metadata 1.21, whose generated protobuf code
# needs a 6.x runtime, while this TensorFlow line pins protobuf below 6. Importing
# dlimp then dies on "gencode 6.31.1 runtime 5.29.6". requirements-data.txt
# already pins the compatible pair, so apply it before verifying the imports
# rather than after. Idempotent when a caller installs it again later.
python -m pip install -q -r "$PROJECT_ROOT/training/openvla_oft_a100/requirements-data.txt"

python -c "import tensorflow_graphics.geometry.transformation"
python -c "import dlimp"

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install "flash-attn==2.5.5" --no-build-isolation
fi

# Registering the PARC dataset belongs to the OpenVLA-OFT checkout, which is
# rebuilt with every VM, not to dataset conversion. prepare_stage_a_rlds.sh used
# to be the only caller, so restoring an already converted dataset from Drive
# skipped it and training failed with KeyError: 'parc_stage_a_plus_only'.
python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_parc_dataset_registry.py" \
  --openvla-root "$WORKDIR"

python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_finetune_component_init.py" \
  "$WORKDIR/vla-scripts/finetune.py"
python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/verify_training_patch.py" \
  "$WORKDIR/vla-scripts/finetune.py"

# Written last so a run interrupted partway never looks complete.
echo "$OPENVLA_OFT_COMMIT" > "$STAMP"

python - <<'PY'
import json
import subprocess
import torch
import transformers

result = {
    "openvla_oft_commit": subprocess.check_output(
        ["git", "-C", "/content/openvla-oft", "rev-parse", "HEAD"], text=True
    ).strip(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available() else None,
}
print(json.dumps(result, indent=2))
PY
