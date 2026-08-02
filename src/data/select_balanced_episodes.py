from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _evenly_spaced_indices(size: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count > size:
        raise ValueError(f"Requested {count} samples from a group of {size}")
    if count == 1:
        return [size // 2]
    return [round(i * (size - 1) / (count - 1)) for i in range(count)]


def _stable_jitter(rows: list[dict[str, Any]], seed: int, task_key: str) -> list[dict[str, Any]]:
    digest = hashlib.sha256(f"{seed}:{task_key}".encode()).digest()
    task_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(task_seed)
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[int(row["length"])].append(row)
    result: list[dict[str, Any]] = []
    for length in sorted(by_length):
        tied = sorted(by_length[length], key=lambda item: int(item["episode_index"]))
        rng.shuffle(tied)
        result.extend(tied)
    return result


def select(inventory: dict[str, Any], per_task: int, train_per_task: int, seed: int) -> dict[str, Any]:
    if not 0 < train_per_task < per_task:
        raise ValueError("train_per_task must be between 1 and per_task - 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in inventory["episodes"]:
        groups[episode["instruction"]].append(episode)
    if len(groups) != 40:
        raise ValueError(f"Expected 40 task groups, found {len(groups)}")

    selected: list[dict[str, Any]] = []
    for instruction, rows in sorted(groups.items()):
        ordered = _stable_jitter(rows, seed, instruction)
        positions = _evenly_spaced_indices(len(ordered), per_task)
        chosen = [ordered[position] for position in positions]
        validation_count = per_task - train_per_task
        validation_positions = set(_evenly_spaced_indices(per_task, validation_count))
        for position, row in enumerate(chosen):
            selected.append(
                {
                    **row,
                    "split": "validation" if position in validation_positions else "train",
                    "selection_reason": "task_balanced_episode_length_quantiles",
                }
            )

    split_counts = Counter(row["split"] for row in selected)
    suite_counts = Counter((row["split"], row["suite"]) for row in selected)
    train_ids = {row["episode_index"] for row in selected if row["split"] == "train"}
    validation_ids = {row["episode_index"] for row in selected if row["split"] == "validation"}
    if train_ids & validation_ids:
        raise AssertionError("Train and validation episode overlap")

    return {
        "selection_version": "1.0.0",
        "source": inventory["source"],
        "strategy": {
            "name": "task_balanced_episode_length_quantiles",
            "seed": seed,
            "per_task": per_task,
            "train_per_task": train_per_task,
            "validation_per_task": per_task - train_per_task,
            "limitations": [
                "This metadata-only strategy balances task, suite, and trajectory length.",
                "It does not claim to balance perturbation categories because those labels are absent.",
                "After candidate parquet download, replace or compare with initial-state farthest-point sampling.",
            ],
        },
        "counts": {
            "total": len(selected),
            "by_split": dict(split_counts),
            "by_split_and_suite": {
                split: {suite: suite_counts[(split, suite)] for suite in ("spatial", "object", "goal", "long")}
                for split in ("train", "validation")
            },
        },
        "episodes": sorted(selected, key=lambda row: (row["split"], row["suite"], row["instruction"], row["episode_index"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a deterministic task-balanced LeRobot episode subset")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    payload = select(inventory, args.per_task, args.train_per_task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
