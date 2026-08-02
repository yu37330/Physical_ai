from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_initial_state(path: Path) -> list[float]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for initial-state refinement") from exc
    table = pq.read_table(path, columns=["observation.state"])
    if table.num_rows == 0:
        raise ValueError(f"Episode parquet has no rows: {path}")
    value = table.column(0)[0].as_py()
    state = [float(item) for item in value]
    if len(state) != 8 or not all(math.isfinite(item) for item in state):
        raise ValueError(f"Expected finite 8D initial state in {path}, got {state}")
    return state


def _standardize(vectors: list[list[float]]) -> list[list[float]]:
    dims = len(vectors[0])
    means = [sum(row[d] for row in vectors) / len(vectors) for d in range(dims)]
    stds = []
    for d in range(dims):
        variance = sum((row[d] - means[d]) ** 2 for row in vectors) / len(vectors)
        stds.append(max(math.sqrt(variance), 1e-6))
    return [[(row[d] - means[d]) / stds[d] for d in range(dims)] for row in vectors]


def _squared_distance(a: list[float], b: list[float]) -> float:
    return sum((left - right) ** 2 for left, right in zip(a, b, strict=True))


def _farthest_point_indices(vectors: list[list[float]], target: int) -> list[int]:
    if target > len(vectors):
        raise ValueError(f"Cannot select {target} from {len(vectors)} vectors")
    standardized = _standardize(vectors)
    centroid = [sum(row[d] for row in standardized) / len(standardized) for d in range(len(standardized[0]))]
    selected = [max(range(len(standardized)), key=lambda index: _squared_distance(standardized[index], centroid))]
    min_distances = [_squared_distance(row, standardized[selected[0]]) for row in standardized]
    while len(selected) < target:
        next_index = max(
            (index for index in range(len(standardized)) if index not in selected),
            key=lambda index: (min_distances[index], -index),
        )
        selected.append(next_index)
        for index, row in enumerate(standardized):
            min_distances[index] = min(min_distances[index], _squared_distance(row, standardized[next_index]))
    return selected


def refine(selection: dict[str, Any], dataset_root: Path, train_per_task: int, validation_per_task: int) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selection["episodes"]:
        episode_index = int(row["episode_index"])
        chunk = episode_index // 1000
        parquet = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        enriched = {**row, "initial_state": _read_initial_state(parquet)}
        groups[(row["instruction"], row["split"])].append(enriched)

    final: list[dict[str, Any]] = []
    for (instruction, split), rows in sorted(groups.items()):
        target = train_per_task if split == "train" else validation_per_task
        rows = sorted(rows, key=lambda row: int(row["episode_index"]))
        indices = _farthest_point_indices([row["initial_state"] for row in rows], target)
        for index in indices:
            final.append({**rows[index], "selection_reason": "initial_state_farthest_point"})

    counts = Counter(row["split"] for row in final)
    return {
        "selection_version": "1.1.0",
        "source": selection["source"],
        "strategy": {
            "name": "task_split_balanced_initial_state_farthest_point",
            "candidate_strategy": selection["strategy"],
            "train_per_task": train_per_task,
            "validation_per_task": validation_per_task,
            "distance_space": "per_task_zscored_8d_initial_state",
            "limitations": [
                "Initial-state diversity is a proxy for robot perturbation diversity.",
                "Camera, layout, light, background, language, and noise categories remain unknown.",
            ],
        },
        "counts": {"total": len(final), "by_split": dict(counts)},
        "episodes": sorted(final, key=lambda row: (row["split"], row["suite"], row["instruction"], row["episode_index"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine task-balanced candidates using initial-state farthest-point sampling")
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--validation-per-task", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.candidate_selection.read_text(encoding="utf-8"))
    payload = refine(selection, args.dataset_root, args.train_per_task, args.validation_per_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
