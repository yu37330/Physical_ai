from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.convert_selected_lerobot_to_rlds import _episode_paths, _read_vectors, _read_video
from src.data.rlds_contract import DATASET_NAME, DATASET_VERSION


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _scalar_int(value: Any) -> int:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return int(np.asarray(value).item())


def _stack_steps(steps_dataset: Any) -> dict[str, np.ndarray]:
    steps = list(steps_dataset.as_numpy_iterator())
    if not steps:
        raise ValueError("RLDS episode contains no steps")
    return {
        "state": np.stack([step["observation"]["state"] for step in steps]),
        "action": np.stack([step["action"] for step in steps]),
        "front": np.stack([step["observation"]["image"] for step in steps]),
        "wrist": np.stack([step["observation"]["wrist_image"] for step in steps]),
    }


def _image_report(source: np.ndarray, converted: np.ndarray) -> dict[str, float | bool]:
    source_f = source.astype(np.float32)
    converted_f = converted.astype(np.float32)
    direct_mad = float(np.mean(np.abs(source_f - converted_f)))
    rotated_mad = float(np.mean(np.abs(np.rot90(source_f, 2, axes=(1, 2)) - converted_f)))
    return {
        "direct_mean_abs_error": direct_mad,
        "rotated_mean_abs_error": rotated_mad,
        "orientation_preserved": direct_mad <= rotated_mad,
        "pixel_parity_within_tolerance": direct_mad <= 1.0,
    }


def validate_episode(
    *,
    source_root: Path,
    episode: dict[str, Any],
    front_video_key: str,
    wrist_video_key: str,
    state_column: str,
    action_column: str,
    image_size: int,
) -> dict[str, Any]:
    metadata = episode["episode_metadata"]
    episode_index = _scalar_int(metadata["source_episode_index"])
    paths = _episode_paths(source_root, episode_index, front_video_key, wrist_video_key)
    source_state, source_action = _read_vectors(paths["parquet"], state_column, action_column)
    source_front = _read_video(paths["front"], image_size=image_size, rotate_180=False)
    source_wrist = _read_video(paths["wrist"], image_size=image_size, rotate_180=False)
    converted = _stack_steps(episode["steps"])

    lengths = {
        "source_state": len(source_state),
        "source_action": len(source_action),
        "source_front": len(source_front),
        "source_wrist": len(source_wrist),
        "rlds_state": len(converted["state"]),
        "rlds_action": len(converted["action"]),
        "rlds_front": len(converted["front"]),
        "rlds_wrist": len(converted["wrist"]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode {episode_index} length mismatch: {lengths}")

    state_max_error = float(np.max(np.abs(source_state - converted["state"])))
    action_max_error = float(np.max(np.abs(source_action - converted["action"])))
    front_report = _image_report(source_front, converted["front"])
    wrist_report = _image_report(source_wrist, converted["wrist"])

    passed = (
        state_max_error <= 1e-6
        and action_max_error <= 1e-6
        and bool(front_report["orientation_preserved"])
        and bool(wrist_report["orientation_preserved"])
        and bool(front_report["pixel_parity_within_tolerance"])
        and bool(wrist_report["pixel_parity_within_tolerance"])
    )
    return {
        "episode_index": episode_index,
        "episode_id": _decode_text(metadata["episode_id"]),
        "frame_count": lengths["rlds_state"],
        "state_max_abs_error": state_max_error,
        "action_max_abs_error": action_max_error,
        "front": front_report,
        "wrist": wrist_report,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate selected LeRobot payload against raw converted RLDS episodes")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tfds-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--episodes-per-split", type=int, default=2)
    parser.add_argument("--front-video-key", default="observation.images.front")
    parser.add_argument("--wrist-video-key", default="observation.images.wrist")
    parser.add_argument("--state-column", default="observation.state")
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError("tensorflow-datasets is required for source parity validation") from exc

    builder_dir = args.tfds_root / args.dataset_name / args.dataset_version
    builder = tfds.builder_from_directory(str(builder_dir))
    split_reports: list[dict[str, Any]] = []
    for split in ("train", "val"):
        dataset = builder.as_dataset(split=split, shuffle_files=False)
        episode_reports: list[dict[str, Any]] = []
        for episode in dataset.take(args.episodes_per_split):
            report = validate_episode(
                source_root=args.source_root,
                episode=episode,
                front_video_key=args.front_video_key,
                wrist_video_key=args.wrist_video_key,
                state_column=args.state_column,
                action_column=args.action_column,
                image_size=args.image_size,
            )
            episode_reports.append(report)
        if len(episode_reports) != args.episodes_per_split:
            raise ValueError(
                f"Requested {args.episodes_per_split} episodes from {split}, received {len(episode_reports)}"
            )
        split_reports.append({"split": split, "episodes": episode_reports})

    all_episodes = [episode for split in split_reports for episode in split["episodes"]]
    status = "pass" if all(episode["passed"] for episode in all_episodes) else "fail"
    payload = {
        "status": status,
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "checks": {
            "state_raw_parity": all(episode["state_max_abs_error"] <= 1e-6 for episode in all_episodes),
            "action_raw_parity": all(episode["action_max_abs_error"] <= 1e-6 for episode in all_episodes),
            "front_orientation_preserved": all(episode["front"]["orientation_preserved"] for episode in all_episodes),
            "wrist_orientation_preserved": all(episode["wrist"]["orientation_preserved"] for episode in all_episodes),
            "front_pixel_parity": all(episode["front"]["pixel_parity_within_tolerance"] for episode in all_episodes),
            "wrist_pixel_parity": all(episode["wrist"]["pixel_parity_within_tolerance"] for episode in all_episodes),
        },
        "splits": split_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, **payload["checks"]}, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
