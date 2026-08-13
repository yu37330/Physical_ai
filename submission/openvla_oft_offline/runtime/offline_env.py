from __future__ import annotations

import os

from .cuda_preload import preload_nvjitlink


def configure_offline_environment() -> None:
    """Disable network-backed model resolution before importing Transformers."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    # torchより先。OpenVLAOfflineRuntime.__init__はこれを最初に呼び、torchが最初に
    # 読み込まれるのはこの後のtransformers importなので、ここが唯一の確実な位置。
    # 理由はcuda_preload.pyを参照。
    preload_nvjitlink()
