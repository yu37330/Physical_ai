from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    config: Path
    model_index: Path
    model_shards: tuple[Path, ...]
    action_head: Path
    proprio_projector: Path
    dataset_statistics: Path


def _find_exactly_one(root: Path, pattern: str) -> Path:
    matches = tuple(sorted(root.glob(pattern)))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file matching {pattern!r} under {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def inspect_checkpoint(root: str | Path) -> CheckpointLayout:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    config = root / "config.json"
    model_index = root / "model.safetensors.index.json"
    dataset_statistics = root / "dataset_statistics.json"

    for required in (config, model_index, dataset_statistics):
        if not required.is_file():
            raise FileNotFoundError(f"Required checkpoint file not found: {required}")

    model_shards = tuple(sorted(root.glob("model-*.safetensors")))
    if not model_shards:
        raise FileNotFoundError(f"No model safetensor shards found under {root}")

    action_head = _find_exactly_one(root, "*action_head*checkpoint*.pt")
    proprio_projector = _find_exactly_one(root, "*proprio_projector*checkpoint*.pt")

    with config.open("r", encoding="utf-8") as handle:
        json.load(handle)
    with dataset_statistics.open("r", encoding="utf-8") as handle:
        json.load(handle)

    return CheckpointLayout(
        root=root,
        config=config,
        model_index=model_index,
        model_shards=model_shards,
        action_head=action_head,
        proprio_projector=proprio_projector,
        dataset_statistics=dataset_statistics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    layout = inspect_checkpoint(args.checkpoint_dir)
    print(json.dumps({
        "root": str(layout.root),
        "model_shards": [str(path) for path in layout.model_shards],
        "action_head": str(layout.action_head),
        "proprio_projector": str(layout.proprio_projector),
        "dataset_statistics": str(layout.dataset_statistics),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
