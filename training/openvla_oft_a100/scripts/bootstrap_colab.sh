#!/usr/bin/env bash
set -euo pipefail

OPENVLA_OFT_COMMIT="${OPENVLA_OFT_COMMIT:-e4287e94541f459edc4feabc4e181f537cd569a8}"
WORKDIR="${WORKDIR:-/content/openvla-oft}"
PROJECT_ROOT="${PROJECT_ROOT:-/content/Physical_ai}"

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
python -c "import tensorflow_graphics.geometry.transformation"
python -c "import dlimp"

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install "flash-attn==2.5.5" --no-build-isolation
fi

python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/patch_finetune_component_init.py" \
  "$WORKDIR/vla-scripts/finetune.py"
python "$PROJECT_ROOT/training/openvla_oft_a100/scripts/verify_training_patch.py" \
  "$WORKDIR/vla-scripts/finetune.py"

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
