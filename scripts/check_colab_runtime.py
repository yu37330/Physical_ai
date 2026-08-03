#!/usr/bin/env python3
"""Fail fast when a Colab notebook is attached to an incompatible runtime."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the active Colab runtime.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless PyTorch can access CUDA and nvidia-smi is available.",
    )
    return parser


def inspect_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    vram_gib = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if cuda_available
        else None
    )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gib": vram_gib,
        "nvidia_smi": shutil.which("nvidia-smi"),
    }


def main() -> None:
    args = build_parser().parse_args()
    report = inspect_runtime()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required.")
    if args.require_cuda:
        if not report["nvidia_smi"]:
            errors.append(
                "nvidia-smi was not found. Reconnect the notebook to a Colab GPU runtime."
            )
        if not report["cuda_available"]:
            errors.append(
                "torch.cuda.is_available() is False. The current kernel is not a GPU runtime."
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
