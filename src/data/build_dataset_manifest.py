from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    inventory: dict[str, Any],
    selection: dict[str, Any],
    dataset_id: str,
    selection_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    episodes = selection["episodes"]
    split_counts = Counter(row["split"] for row in episodes)
    suite_counts = Counter(row["suite"] for row in episodes)
    task_count = len({row["instruction"] for row in episodes})
    selected_ids = [int(row["episode_index"]) for row in episodes]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selection contains duplicate episode indices")

    features = inventory["declared"].get("features", {})
    image_features = [
        {"key": key, "shape": value.get("shape"), "dtype": value.get("dtype")}
        for key, value in features.items()
        if key.startswith("observation.images.")
    ]
    state_feature = features.get("observation.state", {})
    action_feature = features.get("action", {})

    return {
        "manifest_version": "2.0.0",
        "dataset_id": dataset_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "PARC2026 Stage A minimal adaptation of OpenVLA-OFT+ Action Head and Proprio Projector",
        "provenance": {
            "sources": [inventory["source"]],
            "inventory_file": str(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "selection_file": str(selection_path),
            "selection_sha256": _sha256(selection_path),
        },
        "compliance": {
            "allowed_for_training": True,
            "contains_official_evaluation_data": False,
            "official_evaluation_data_segregated": True,
            "generation_method": "public_dataset_subset",
            "disclosure_notes": [
                "Episode selection uses only public training metadata.",
                "Official PARC evaluation observations, seeds, and private tasks are excluded.",
            ],
        },
        "structure": {
            "format": "lerobot_subset_pending_rlds_conversion",
            "robot_type": inventory["declared"].get("robot_type"),
            "fps": inventory["declared"].get("fps"),
            "task_count": task_count,
            "episode_count": len(episodes),
            "frame_count": sum(int(row["length"]) for row in episodes),
            "modalities": {
                "images": image_features,
                "state": {
                    "key": "observation.state",
                    "shape": state_feature.get("shape"),
                    "dtype": state_feature.get("dtype"),
                    "semantic_order": ["eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper_0", "gripper_1"],
                },
                "action": {
                    "key": "action",
                    "shape": action_feature.get("shape"),
                    "dtype": action_feature.get("dtype"),
                    "semantic_order": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
                },
            },
        },
        "taxonomy": {
            "suite_episode_counts": {suite: suite_counts[suite] for suite in ("spatial", "object", "goal", "long")},
            "perturbation_labels": {
                "status": "unknown",
                "label_source": "not_in_public_lerobot_metadata",
                "counts": {"unknown": len(episodes)},
            },
        },
        "selection": {
            "unit": "episode",
            "strategy": selection["strategy"],
            "selected_episode_count": len(episodes),
        },
        "splits": {
            split: {
                "episode_count": split_counts[split],
                "episode_indices": sorted(int(row["episode_index"]) for row in episodes if row["split"] == split),
                "split_strategy": "episode_level_task_balanced",
            }
            for split in ("train", "validation")
        },
        "transformations": [
            {"name": "rotate_image_180", "version": "1", "deterministic": True, "parameters": {"camera_keys": ["front", "wrist"]}},
            {"name": "parc_resolution_simulation", "version": "1", "deterministic": True, "parameters": {"downsample_to": [128, 128]}},
            {"name": "convert_lerobot_to_rlds", "version": "pending", "deterministic": True, "parameters": {"action_chunk_length": 8}},
        ],
        "quality": {
            "checks": inventory["checks"],
            "train_validation_overlap": False,
            "status": "metadata_selected_pending_payload_validation",
        },
        "limitations": list(dict.fromkeys(inventory.get("limitations", []) + selection["strategy"].get("limitations", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned PARC dataset manifest")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    manifest = build_manifest(inventory, selection, args.dataset_id, args.selection, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "episodes": manifest["structure"]["episode_count"]}, indent=2))


if __name__ == "__main__":
    main()
