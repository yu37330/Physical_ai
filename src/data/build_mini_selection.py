from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mini_selection(
    selection: dict[str, Any],
    *,
    instruction: str | None,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    if train_count < 1 or validation_count < 1:
        raise ValueError("train_count and validation_count must both be positive")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "validation": []}
    )
    for row in selection.get("episodes", []):
        split = str(row.get("split"))
        if split not in ("train", "validation"):
            continue
        grouped[str(row["instruction"])][split].append(row)

    if instruction is None:
        eligible = [
            task
            for task, splits in grouped.items()
            if len(splits["train"]) >= train_count
            and len(splits["validation"]) >= validation_count
        ]
        if not eligible:
            raise ValueError("No task has enough train and validation episodes for the mini selection")
        instruction = sorted(eligible)[0]
    elif instruction not in grouped:
        raise KeyError(f"Instruction not found in selection: {instruction}")

    splits = grouped[instruction]
    if len(splits["train"]) < train_count or len(splits["validation"]) < validation_count:
        raise ValueError(
            f"Task {instruction!r} has train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}; requested {train_count}/{validation_count}"
        )

    train_rows = sorted(splits["train"], key=lambda row: int(row["episode_index"]))[:train_count]
    validation_rows = sorted(
        splits["validation"], key=lambda row: int(row["episode_index"])
    )[:validation_count]
    episodes = [*train_rows, *validation_rows]

    return {
        "selection_version": "mini-1.0.0",
        "source": selection.get("source", {}),
        "parent_selection": {
            "selection_version": selection.get("selection_version"),
            "strategy": selection.get("strategy", {}),
        },
        "strategy": {
            "name": "single_task_deterministic_mini_e2e",
            "instruction": instruction,
            "train_count": train_count,
            "validation_count": validation_count,
            "ordering": "episode_index_ascending",
        },
        "counts": {
            "total": len(episodes),
            "by_split": {"train": len(train_rows), "validation": len(validation_rows)},
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic 3-episode E2E selection")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    payload = build_mini_selection(
        selection,
        instruction=args.instruction,
        train_count=args.train_count,
        validation_count=args.validation_count,
    )
    payload["parent_selection_sha256"] = _sha256(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(payload["strategy"]["instruction"])


if __name__ == "__main__":
    main()
