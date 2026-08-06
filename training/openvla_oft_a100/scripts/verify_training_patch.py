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
    "FrozenModuleWrapper",
    "base VLA frozen without adding random LoRA weights",
    "vision-backbone LoRA parameters frozen",
    "if cfg.use_lora and cfg.train_vla_lora",
    "PARC_BATCH_DEVICE_PATCH",
)

FORBIDDEN_SNIPPETS = (
    # labels on CPU makes the action masks CPU tensors, which the model then
    # multiplies against CUDA embeddings.
    '            labels=batch["labels"],\n',
    '                labels=batch["labels"],\n',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finetune_py", type=Path)
    args = parser.parse_args()

    source = args.finetune_py.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in source]
    if missing:
        raise SystemExit(f"Training patch is incomplete; missing: {missing}")
    remaining = [snippet for snippet in FORBIDDEN_SNIPPETS if snippet in source]
    if remaining:
        raise SystemExit(
            f"Training patch left an unpatched forward call: {remaining!r}"
        )
    py_compile.compile(str(args.finetune_py), doraise=True)
    print(f"Training patch verified: {args.finetune_py}")


if __name__ == "__main__":
    main()
