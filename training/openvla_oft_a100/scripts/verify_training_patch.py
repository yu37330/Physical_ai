from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

REQUIRED_SNIPPETS = (
    "PARC_COMPONENT_CHECKPOINT_PATCH",
    "component_checkpoint_dir",
    "train_vla_lora",
    "freeze_vision_lora",
    "Loaded pretrained {module_name}",
    "PARC Stage A",
    "PARC Stage B",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finetune_py", type=Path)
    args = parser.parse_args()

    source = args.finetune_py.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in source]
    if missing:
        raise SystemExit(f"Training patch is incomplete; missing: {missing}")
    py_compile.compile(str(args.finetune_py), doraise=True)
    print(f"Training patch verified: {args.finetune_py}")


if __name__ == "__main__":
    main()
