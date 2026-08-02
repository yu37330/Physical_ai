#!/usr/bin/env bash
set -euo pipefail

OPENVLA_OFT_COMMIT="${OPENVLA_OFT_COMMIT:-main}"
WORKDIR="${WORKDIR:-/content/openvla-oft}"

git clone https://github.com/moojink/openvla-oft.git "$WORKDIR"
cd "$WORKDIR"
git checkout "$OPENVLA_OFT_COMMIT"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -e .
python -m pip install "flash-attn==2.5.5" --no-build-isolation

python - <<'PY'
import torch
import transformers
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available() else None,
})
PY
