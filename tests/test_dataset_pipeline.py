from __future__ import annotations

import json
from pathlib import Path

# Import as package members, the way `python -m src.data.<name>` does. Putting
# src/data on sys.path instead let bare sibling imports pass here while failing
# on the real command line.
from src.data.build_dataset_manifest import build_manifest
from src.data.inspect_lerobot_metadata import build_inventory
from src.data.libero_taxonomy import load_suite_map
from src.data.select_balanced_episodes import select

ROOT = Path(__file__).resolve().parents[1]


def _make_meta(tmp_path: Path, suite_map_path: Path) -> Path:
    suite_map = json.loads(suite_map_path.read_text(encoding="utf-8"))["suites"]
    tasks = [task for suite in ("spatial", "object", "goal", "long") for task in suite_map[suite]]
    meta = tmp_path / "meta"
    meta.mkdir()
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": 800,
        "total_frames": 83600,
        "total_tasks": 40,
        "fps": 20,
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [256, 256, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            handle.write(json.dumps({"task_index": index, "task": task}) + "\n")
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        episode_index = 0
        total = 0
        for task in tasks:
            for offset in range(20):
                length = 95 + offset
                total += length
                handle.write(json.dumps({"episode_index": episode_index, "tasks": [task], "length": length}) + "\n")
                episode_index += 1
    info["total_frames"] = total
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return meta


def test_taxonomy_has_40_unique_tasks() -> None:
    mapping = load_suite_map(ROOT / "configs" / "datasets" / "libero_task_suite_map.json")
    assert len(mapping) == 40
    assert set(mapping.values()) == {"spatial", "object", "goal", "long"}


def test_inventory_selection_and_manifest(tmp_path: Path) -> None:
    suite_map_path = ROOT / "configs" / "datasets" / "libero_task_suite_map.json"
    meta = _make_meta(tmp_path, suite_map_path)
    source = {
        "repo_id": "example/libero_plus",
        "repo_type": "dataset",
        "resolved_revision": "a" * 40,
        "license": "mit",
        "metadata_root": str(meta),
    }
    inventory = build_inventory(meta, suite_map_path, source)
    assert all(inventory["checks"][key] for key in (
        "declared_episode_count_matches",
        "declared_task_count_matches",
        "declared_frame_count_matches",
        "all_tasks_classified",
    ))
    selection = select(inventory, per_task=20, train_per_task=16, seed=20260802)
    assert selection["counts"]["by_split"] == {"train": 640, "validation": 160}
    for split in ("train", "validation"):
        suite_counts = selection["counts"]["by_split_and_suite"][split]
        assert len(set(suite_counts.values())) == 1

    inventory_path = tmp_path / "inventory.json"
    selection_path = tmp_path / "selection.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    manifest = build_manifest(inventory, selection, "parc_stage_a_balanced_v001", selection_path, inventory_path)
    assert manifest["structure"]["episode_count"] == 800
    assert manifest["compliance"]["contains_official_evaluation_data"] is False
    assert manifest["quality"]["train_validation_overlap"] is False

    import jsonschema
    schema = json.loads((ROOT / "schemas" / "dataset_manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(manifest, schema)
