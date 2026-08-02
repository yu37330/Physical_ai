from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> None:
    output = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "packages": {
            name: package_version(name)
            for name in (
                "transformers",
                "tokenizers",
                "peft",
                "accelerate",
                "timm",
                "flash-attn",
                "tensorflow",
            )
        },
        "git": {},
    }

    for repo in (Path.cwd(), Path.cwd().parent):
        if (repo / ".git").exists():
            output["git"][str(repo)] = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
