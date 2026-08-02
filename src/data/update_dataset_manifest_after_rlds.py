from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _upsert_transformation(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    for index, current in enumerate(items):
        if current.get("name") == item["name"]:
            items[index] = item
            return
    items.append(item)


def update_manifest(
    manifest: dict[str, Any],
    conversion: dict[str, Any],
    parity: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    if parity.get("status") != "pass":
        raise ValueError("Source-to-RLDS parity report did not pass")
    if compatibility.get("status") != "pass":
        raise ValueError("OpenVLA compatibility report did not pass")

    parity_required = (
        "state_raw_parity",
        "action_raw_parity",
        "front_orientation_preserved",
        "wrist_orientation_preserved",
        "front_pixel_parity",
        "wrist_pixel_parity",
    )
    parity_failed = [name for name in parity_required if not parity.get("checks", {}).get(name)]
    if parity_failed:
        raise ValueError(f"Parity report has failed or missing checks: {parity_failed}")

    compatibility_required = (
        "rlds_dataset_constructed",
        "rlds_batch_transform_passed",
        "collator_passed",
        "front_and_wrist_tensors_present",
        "finite_action_and_proprio",
        "action_chunk_shape_passed",
    )
    compatibility_failed = [
        name for name in compatibility_required if not compatibility.get("checks", {}).get(name)
    ]
    if compatibility_failed:
        raise ValueError(
            f"Compatibility report has failed or missing checks: {compatibility_failed}"
        )

    manifest["structure"]["format"] = "tfds_rlds"
    manifest["structure"]["episode_count"] = sum(conversion["episode_counts"].values())
    if conversion.get("frame_counts"):
        manifest["structure"]["frame_count"] = sum(conversion["frame_counts"].values())

    _upsert_transformation(
        manifest["transformations"],
        {
            "name": "selected_lerobot_to_tfds_rlds",
            "version": conversion["version"],
            "deterministic": True,
            "parameters": {
                "dataset_name": conversion["dataset_name"],
                "selection_sha256": conversion["selection_sha256"],
                "state_dim": conversion["contract"]["state_dim"],
                "action_dim": conversion["contract"]["action_dim"],
                "image_size": conversion["contract"]["image_size"],
                "tfds_splits": conversion.get("tfds_splits", ["train", "val"]),
                "rotate_180_during_conversion": conversion["contract"]["rotate_180_during_conversion"],
                "source_parity_checks": parity["checks"],
            },
        },
    )
    _upsert_transformation(
        manifest["transformations"],
        {
            "name": "openvla_oft_rlds_batch_transform",
            "version": "pinned_openvla_oft",
            "deterministic": True,
            "parameters": compatibility["contract"],
        },
    )

    checks = manifest["quality"].setdefault("checks", {})
    checks.update(
        {
            "rlds_conversion_completed": True,
            "state_raw_parity": parity["checks"]["state_raw_parity"],
            "action_raw_parity": parity["checks"]["action_raw_parity"],
            "front_orientation_preserved": parity["checks"]["front_orientation_preserved"],
            "wrist_orientation_preserved": parity["checks"]["wrist_orientation_preserved"],
            "rlds_dataset_constructed": compatibility["checks"]["rlds_dataset_constructed"],
            "rlds_batch_transform_passed": compatibility["checks"]["rlds_batch_transform_passed"],
            "openvla_collator_passed": compatibility["checks"]["collator_passed"],
            "front_and_wrist_tensors_present": compatibility["checks"]["front_and_wrist_tensors_present"],
            "action_chunk_shape_passed": compatibility["checks"]["action_chunk_shape_passed"],
            "finite_action_and_proprio": compatibility["checks"]["finite_action_and_proprio"],
        }
    )
    manifest["quality"]["status"] = "payload_validated"
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote Dataset Manifest after RLDS/OpenVLA compatibility validation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    conversion = json.loads(args.conversion_report.read_text(encoding="utf-8"))
    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    compatibility = json.loads(args.compatibility_report.read_text(encoding="utf-8"))
    result = update_manifest(manifest, conversion, parity, compatibility)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_id": result["dataset_id"], "status": result["quality"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
