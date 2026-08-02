from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_parquet(path: Path, states: np.ndarray, actions: np.ndarray) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to generate the synthetic fixture") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), states.shape[1])
            ),
            "action": pa.array(
                actions.tolist(), type=pa.list_(pa.float32(), actions.shape[1])
            ),
        }
    )
    pq.write_table(table, path)


def _frame(image_size: int, episode_index: int, timestep: int, wrist: bool) -> np.ndarray:
    y, x = np.mgrid[0:image_size, 0:image_size]
    offset = 71 if wrist else 13
    red = (x * 5 + episode_index * 31 + offset) % 256
    green = (y * 7 + timestep * 29 + offset * 2) % 256
    blue = ((x + y) * 3 + episode_index * 17 + timestep * 11) % 256
    image = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    # Add a directional marker so a 180-degree error is easy to detect.
    marker = max(2, image_size // 8)
    image[:marker, : marker * 2] = np.asarray([255, 16, 16], dtype=np.uint8)
    image[-marker:, -marker * 2 :] = np.asarray([16, 16, 255], dtype=np.uint8)
    return image


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required to generate MP4 fixtures") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 fixture: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def generate_fixture(
    root: Path,
    selection_path: Path,
    *,
    frame_count: int = 8,
    image_size: int = 32,
    fps: int = 20,
) -> dict[str, Any]:
    if frame_count < 8:
        raise ValueError("frame_count must be at least 8 to validate an 8-step action chunk")
    if image_size < 16 or image_size % 2:
        raise ValueError("image_size must be an even integer of at least 16")

    instruction = "pick up the synthetic bowl and place it on the synthetic plate"
    episodes: list[dict[str, Any]] = []
    split_by_episode = {0: "train", 1: "train", 2: "validation"}

    for episode_index, split in split_by_episode.items():
        t = np.arange(frame_count, dtype=np.float32)
        states = np.stack(
            [
                0.10 + episode_index * 0.01 + t * 0.001,
                -0.20 + t * 0.002,
                0.30 - t * 0.001,
                t * 0.003,
                -t * 0.002,
                t * 0.001,
                np.full_like(t, 0.02 + episode_index * 0.001),
                np.full_like(t, 0.02 + episode_index * 0.001),
            ],
            axis=1,
        ).astype(np.float32)
        actions = np.stack(
            [
                np.sin(t / 5.0) * 0.01,
                np.cos(t / 5.0) * 0.01,
                np.full_like(t, 0.002),
                np.zeros_like(t),
                np.zeros_like(t),
                np.zeros_like(t),
                np.where(t < frame_count // 2, -1.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        chunk = episode_index // 1000
        parquet_path = root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        _write_parquet(parquet_path, states, actions)

        front_frames = [
            _frame(image_size, episode_index, timestep, wrist=False)
            for timestep in range(frame_count)
        ]
        wrist_frames = [
            _frame(image_size, episode_index, timestep, wrist=True)
            for timestep in range(frame_count)
        ]
        _write_mp4(
            root
            / f"videos/chunk-{chunk:03d}/observation.images.front/episode_{episode_index:06d}.mp4",
            front_frames,
            fps,
        )
        _write_mp4(
            root
            / f"videos/chunk-{chunk:03d}/observation.images.wrist/episode_{episode_index:06d}.mp4",
            wrist_frames,
            fps,
        )

        episodes.append(
            {
                "episode_index": episode_index,
                "instruction": instruction,
                "suite": "spatial",
                "split": split,
                "length": frame_count,
                "selection_reason": "synthetic_e2e_fixture",
            }
        )

    selection = {
        "selection_version": "synthetic-1.0.0",
        "source": {
            "repo_id": "local/synthetic_lerobot_fixture",
            "resolved_revision": "synthetic-v1",
            "license": "test-only",
        },
        "strategy": {
            "name": "synthetic_three_episode_e2e",
            "train_per_task": 2,
            "validation_per_task": 1,
        },
        "counts": {"total": 3, "by_split": {"train": 2, "validation": 1}},
        "episodes": episodes,
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic LeRobot-style fixture")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    payload = generate_fixture(
        args.root,
        args.selection,
        frame_count=args.frame_count,
        image_size=args.image_size,
        fps=args.fps,
    )
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
