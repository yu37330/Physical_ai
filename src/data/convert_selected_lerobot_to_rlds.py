from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.data.rlds_contract import (
    ACTION_DIM,
    DATASET_NAME,
    DATASET_VERSION,
    STATE_DIM,
    build_rlds_steps,
    validate_episode_arrays,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_paths(root: Path, episode_index: int, front_video_key: str, wrist_video_key: str) -> dict[str, Path]:
    chunk = episode_index // 1000
    return {
        "parquet": root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
        "front": root / f"videos/chunk-{chunk:03d}/{front_video_key}/episode_{episode_index:06d}.mp4",
        "wrist": root / f"videos/chunk-{chunk:03d}/{wrist_video_key}/episode_{episode_index:06d}.mp4",
    }


def _read_vectors(path: Path, state_column: str, action_column: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for LeRobot parquet conversion") from exc

    table = pq.read_table(path, columns=[state_column, action_column])
    states = np.asarray(table[state_column].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[action_column].to_pylist(), dtype=np.float32)
    return states, actions


def _read_video(path: Path, *, image_size: int, rotate_180: bool) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video conversion") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate_180:
                frame = np.rot90(frame, 2).copy()
            if frame.shape[:2] != (image_size, image_size):
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.uint8, copy=False))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return np.stack(frames, axis=0)


def load_episode(
    *,
    source_root: Path,
    row: dict[str, Any],
    front_video_key: str,
    wrist_video_key: str,
    state_column: str,
    action_column: str,
    image_size: int,
    rotate_180: bool,
) -> dict[str, Any]:
    episode_index = int(row["episode_index"])
    paths = _episode_paths(source_root, episode_index, front_video_key, wrist_video_key)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Episode {episode_index} is missing files: {missing}")

    states, actions = _read_vectors(paths["parquet"], state_column, action_column)
    front_images = _read_video(paths["front"], image_size=image_size, rotate_180=rotate_180)
    wrist_images = _read_video(paths["wrist"], image_size=image_size, rotate_180=rotate_180)
    contract = validate_episode_arrays(
        states=states,
        actions=actions,
        front_images=front_images,
        wrist_images=wrist_images,
    )
    steps = build_rlds_steps(
        states=states,
        actions=actions,
        front_images=front_images,
        wrist_images=wrist_images,
        instruction=str(row["instruction"]),
    )
    return {
        "steps": steps,
        "episode_metadata": {
            "episode_id": f"episode_{episode_index:06d}",
            "source_episode_index": np.int64(episode_index),
            "suite": str(row["suite"]),
            "task": str(row["instruction"]),
            "source_parquet_sha256": _sha256(paths["parquet"]),
        },
        "contract": asdict(contract),
    }


def _build_tfds(
    *,
    source_root: Path,
    selection: dict[str, Any],
    output_root: Path,
    front_video_key: str,
    wrist_video_key: str,
    state_column: str,
    action_column: str,
    image_size: int,
    rotate_180: bool,
) -> dict[str, Any]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError("tensorflow and tensorflow-datasets are required for RLDS conversion") from exc

    episodes_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for row in selection["episodes"]:
        source_split = str(row["split"])
        tfds_split = "val" if source_split == "validation" else source_split
        if tfds_split not in episodes_by_split:
            raise ValueError(f"Unsupported split {source_split!r}; expected train or validation")
        episodes_by_split[tfds_split].append(row)

    generated_frame_counts = {"train": 0, "val": 0}

    class ParcLiberoPlusSelected(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(DATASET_VERSION)
        RELEASE_NOTES = {DATASET_VERSION: "Selected PARC2026 LIBERO-plus episodes converted from LeRobot."}

        def _info(self) -> tfds.core.DatasetInfo:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": {
                                    "image": tfds.features.Image(
                                        shape=(image_size, image_size, 3), dtype=np.uint8, encoding_format="png"
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(image_size, image_size, 3), dtype=np.uint8, encoding_format="png"
                                    ),
                                    "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                                },
                                "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                                "discount": np.float32,
                                "reward": np.float32,
                                "is_first": np.bool_,
                                "is_last": np.bool_,
                                "is_terminal": np.bool_,
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": {
                            "episode_id": tfds.features.Text(),
                            "source_episode_index": np.int64,
                            "suite": tfds.features.Text(),
                            "task": tfds.features.Text(),
                            "source_parquet_sha256": tfds.features.Text(),
                        },
                    }
                ),
                supervised_keys=None,
                homepage="https://github.com/yu37330/Physical_ai",
            )

        def _split_generators(self, dl_manager: tfds.download.DownloadManager):
            del dl_manager
            return {
                "train": self._generate_examples("train"),
                "val": self._generate_examples("val"),
            }

        def _generate_examples(self, split: str) -> Iterator[tuple[str, dict[str, Any]]]:
            for row in sorted(episodes_by_split[split], key=lambda item: int(item["episode_index"])):
                payload = load_episode(
                    source_root=source_root,
                    row=row,
                    front_video_key=front_video_key,
                    wrist_video_key=wrist_video_key,
                    state_column=state_column,
                    action_column=action_column,
                    image_size=image_size,
                    rotate_180=rotate_180,
                )
                contract = payload.pop("contract")
                generated_frame_counts[split] += int(contract["frame_count"])
                key = payload["episode_metadata"]["episode_id"]
                yield key, payload

    builder = ParcLiberoPlusSelected(data_dir=str(output_root))
    builder.download_and_prepare()

    report = {
        "dataset_name": DATASET_NAME,
        "builder_name": builder.name,
        "version": DATASET_VERSION,
        "data_dir": str(builder.data_dir),
        "source": selection.get("source", {}),
        "selection_sha256": None,
        "episode_counts": {
            "train": len(episodes_by_split["train"]),
            "validation": len(episodes_by_split["val"]),
        },
        "frame_counts": {
            "train": generated_frame_counts["train"],
            "validation": generated_frame_counts["val"],
        },
        "tfds_splits": ["train", "val"],
        "contract": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "front_key": "observation.image",
            "wrist_key": "observation.wrist_image",
            "state_key": "observation.state",
            "language_key": "language_instruction",
            "image_size": image_size,
            "rotate_180_during_conversion": rotate_180,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert selected LeRobot LIBERO-plus episodes to TFDS/RLDS")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--front-video-key", default="observation.images.front")
    parser.add_argument("--wrist-video-key", default="observation.images.wrist")
    parser.add_argument("--state-column", default="observation.state")
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rotate-180", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = _build_tfds(
        source_root=args.source_root,
        selection=selection,
        output_root=args.output_root,
        front_video_key=args.front_video_key,
        wrist_video_key=args.wrist_video_key,
        state_column=args.state_column,
        action_column=args.action_column,
        image_size=args.image_size,
        rotate_180=args.rotate_180,
    )
    report["selection_sha256"] = _sha256(args.selection)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
