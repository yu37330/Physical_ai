from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from libero_taxonomy import classify_instruction, load_suite_map, normalize_instruction


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _read_parquet_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inspect LeRobot v3 parquet metadata") from exc
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _tasks_from_metadata(meta_dir: Path) -> list[dict[str, Any]]:
    jsonl = meta_dir / "tasks.jsonl"
    if jsonl.is_file():
        return _read_jsonl(jsonl)
    parquet = meta_dir / "tasks.parquet"
    if parquet.is_file():
        return _read_parquet_rows([parquet])
    raise FileNotFoundError("Expected meta/tasks.jsonl or meta/tasks.parquet")


def _episodes_from_metadata(meta_dir: Path) -> list[dict[str, Any]]:
    jsonl = meta_dir / "episodes.jsonl"
    if jsonl.is_file():
        return _read_jsonl(jsonl)
    episode_dir = meta_dir / "episodes"
    parquet_paths = list(episode_dir.glob("**/*.parquet")) if episode_dir.is_dir() else []
    if parquet_paths:
        return _read_parquet_rows(parquet_paths)
    raise FileNotFoundError("Expected meta/episodes.jsonl or meta/episodes/**/*.parquet")


def _task_text(row: dict[str, Any]) -> str:
    for key in ("task", "instruction", "task_name"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    tasks = row.get("tasks")
    if isinstance(tasks, list) and len(tasks) == 1 and isinstance(tasks[0], str):
        return tasks[0]
    raise ValueError(f"Cannot determine task text from row keys: {sorted(row)}")


def build_inventory(meta_dir: Path, suite_map_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    info = _read_json(meta_dir / "info.json")
    tasks = _tasks_from_metadata(meta_dir)
    episodes = _episodes_from_metadata(meta_dir)
    suite_map = load_suite_map(suite_map_path)

    canonical_tasks: dict[str, dict[str, Any]] = {}
    for row in tasks:
        instruction = _task_text(row)
        normalized = normalize_instruction(instruction)
        task_index = row.get("task_index")
        canonical_tasks[normalized] = {
            "task_index": int(task_index) if task_index is not None else None,
            "instruction": instruction,
            "suite": classify_instruction(instruction, suite_map),
        }

    episode_rows: list[dict[str, Any]] = []
    counts_by_task: Counter[str] = Counter()
    lengths_by_task: dict[str, list[int]] = defaultdict(list)
    suite_counts: Counter[str] = Counter()
    unknown_tasks: set[str] = set()

    for row in episodes:
        instruction = _task_text(row)
        normalized = normalize_instruction(instruction)
        task = canonical_tasks.get(normalized)
        suite = task["suite"] if task else classify_instruction(instruction, suite_map)
        if suite == "unknown":
            unknown_tasks.add(instruction)
        length = int(row.get("length", row.get("episode_length", 0)))
        episode_index = int(row["episode_index"])
        task_index = task.get("task_index") if task else row.get("task_index")
        counts_by_task[normalized] += 1
        lengths_by_task[normalized].append(length)
        suite_counts[suite] += 1
        episode_rows.append(
            {
                "episode_index": episode_index,
                "task_index": int(task_index) if task_index is not None else None,
                "instruction": instruction,
                "suite": suite,
                "length": length,
                "perturbation": {"category": "unknown", "label_source": "not_in_public_metadata"},
            }
        )

    task_summary: list[dict[str, Any]] = []
    for normalized, count in sorted(counts_by_task.items()):
        lengths = lengths_by_task[normalized]
        task = canonical_tasks.get(normalized, {})
        task_summary.append(
            {
                "task_index": task.get("task_index"),
                "instruction": task.get("instruction", normalized),
                "suite": task.get("suite", "unknown"),
                "episode_count": count,
                "length_min": min(lengths),
                "length_max": max(lengths),
                "length_mean": round(statistics.fmean(lengths), 3),
                "length_median": statistics.median(lengths),
            }
        )

    declared = {
        "task_count": int(info.get("total_tasks", 0)),
        "episode_count": int(info.get("total_episodes", 0)),
        "frame_count": int(info.get("total_frames", 0)),
        "fps": float(info.get("fps", 0)),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "features": info.get("features", {}),
    }
    observed_frames = sum(row["length"] for row in episode_rows)
    checks = {
        "declared_episode_count_matches": declared["episode_count"] == len(episode_rows),
        "declared_task_count_matches": declared["task_count"] == len(task_summary),
        "declared_frame_count_matches": declared["frame_count"] == observed_frames,
        "all_tasks_classified": not unknown_tasks,
        "suite_task_counts": dict(Counter(row["suite"] for row in task_summary)),
    }

    return {
        "inventory_version": "1.0.0",
        "source": source,
        "declared": declared,
        "observed": {
            "task_count": len(task_summary),
            "episode_count": len(episode_rows),
            "frame_count": observed_frames,
            "suite_episode_counts": dict(suite_counts),
            "unknown_tasks": sorted(unknown_tasks),
        },
        "checks": checks,
        "tasks": task_summary,
        "episodes": episode_rows,
        "limitations": [
            "Public metadata does not identify per-episode perturbation categories.",
            "Collision and clearance labels are not present in the public metadata.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a locally downloaded LeRobot metadata directory")
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--suite-map", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "resolved_revision": args.revision,
        "license": args.license,
        "metadata_root": str(args.meta_dir),
    }
    inventory = build_inventory(args.meta_dir, args.suite_map, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": inventory["checks"]}, indent=2))


if __name__ == "__main__":
    main()
