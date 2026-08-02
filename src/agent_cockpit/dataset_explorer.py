"""RLDS Dataset ManifestとEpisodeサンプルをUI向けに読み出す。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.rlds_contract import ACTION_CHUNK_LENGTH, ACTION_DIM, STATE_DIM


@dataclass(frozen=True)
class EpisodeSample:
    """Dataset Explorerと単発推論で共有する1フレーム分のサンプル。"""

    split: str
    episode_offset: int
    frame_id: int
    episode_id: str
    instruction: str
    front_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    action_chunk: np.ndarray
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "episode_offset": self.episode_offset,
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "front_shape": list(self.front_image.shape),
            "wrist_shape": list(self.wrist_image.shape),
            "state_shape": list(self.state.shape),
            "action_shape": list(self.action.shape),
            "action_chunk_shape": list(self.action_chunk.shape),
            "state": self.state.astype(float).tolist(),
            "action": self.action.astype(float).tolist(),
            **self.metadata,
        }


def load_json(path: str | Path) -> dict[str, Any]:
    """JSONを読み込み、UIで扱える辞書として返す。"""

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return payload


def build_dataset_overview(
    *,
    manifest_path: str | Path,
    conversion_report_path: str | Path | None = None,
    compatibility_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Dataset Manifestと変換・互換性レポートを一つの概要へまとめる。"""

    manifest = load_json(manifest_path)
    structure = manifest.get("structure", {})
    quality = manifest.get("quality", {})
    overview: dict[str, Any] = {
        "dataset_id": manifest.get("dataset_id", "unknown"),
        "format": structure.get("format", "unknown"),
        "episode_count": structure.get("episode_count"),
        "frame_count": structure.get("frame_count"),
        "quality_status": quality.get("status", "unknown"),
        "quality_checks": quality.get("checks", {}),
        "allowed_for_training": manifest.get("allowed_for_training"),
        "contains_official_evaluation_data": manifest.get(
            "contains_official_evaluation_data"
        ),
    }

    if conversion_report_path:
        conversion = load_json(conversion_report_path)
        overview["conversion"] = {
            "dataset_name": conversion.get("dataset_name"),
            "version": conversion.get("version"),
            "data_dir": conversion.get("data_dir"),
            "episode_counts": conversion.get("episode_counts", {}),
            "frame_counts": conversion.get("frame_counts", {}),
            "contract": conversion.get("contract", {}),
        }

    if compatibility_report_path:
        compatibility = load_json(compatibility_report_path)
        overview["openvla_compatibility"] = {
            "status": compatibility.get("status"),
            "checks": compatibility.get("checks", {}),
            "contract": compatibility.get("contract", {}),
        }
    return overview


def build_episode_table(selection_path: str | Path) -> list[list[Any]]:
    """Episode選定JSONをGradio Dataframe向けの行へ変換する。"""

    selection = load_json(selection_path)
    episodes = selection.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError("selection.episodes must be a list")
    rows: list[list[Any]] = []
    for row in episodes:
        rows.append(
            [
                int(row.get("episode_index", -1)),
                str(row.get("split", "")),
                str(row.get("suite", "")),
                str(row.get("instruction", "")),
                int(row.get("length", row.get("frame_count", 0)) or 0),
            ]
        )
    return rows


def build_action_chunk(
    actions: Any,
    *,
    frame_id: int,
    chunk_length: int = ACTION_CHUNK_LENGTH,
) -> np.ndarray:
    """現在FrameからAction chunkを切り出し、終端では最終Actionで埋める。"""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions (T, {ACTION_DIM}), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("actions must not be empty")
    if frame_id < 0 or frame_id >= array.shape[0]:
        raise IndexError(f"frame_id {frame_id} is outside [0, {array.shape[0] - 1}]")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")

    chunk = array[frame_id : frame_id + chunk_length]
    if chunk.shape[0] < chunk_length:
        padding = np.repeat(chunk[-1][None, :], chunk_length - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, padding], axis=0)
    return chunk.astype(np.float32, copy=False)


def _decode_text(value: Any) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class RLDSEpisodeReader:
    """変換済みTFDS/RLDS Builder directoryからEpisodeを読み出す。"""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser()
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"RLDS dataset directory not found: {self.dataset_dir}")
        self._builder: Any | None = None

    def _get_builder(self) -> Any:
        if self._builder is None:
            try:
                import tensorflow_datasets as tfds
            except ImportError as exc:
                raise RuntimeError(
                    "tensorflow-datasets is required to read converted RLDS data"
                ) from exc
            self._builder = tfds.builder_from_directory(str(self.dataset_dir))
        return self._builder

    def available_splits(self) -> list[str]:
        builder = self._get_builder()
        return sorted(str(name) for name in builder.info.splits)

    def read_sample(
        self,
        *,
        split: str,
        episode_offset: int,
        frame_id: int,
    ) -> EpisodeSample:
        builder = self._get_builder()
        if split not in builder.info.splits:
            raise KeyError(
                f"Unknown split {split!r}; available={sorted(builder.info.splits)}"
            )
        episode_count = int(builder.info.splits[split].num_examples)
        if episode_offset < 0 or episode_offset >= episode_count:
            raise IndexError(
                f"episode_offset {episode_offset} is outside [0, {episode_count - 1}]"
            )

        dataset = builder.as_dataset(
            split=f"{split}[{episode_offset}:{episode_offset + 1}]",
            shuffle_files=False,
        )
        try:
            episode = next(iter(dataset))
        except StopIteration as exc:
            raise IndexError(
                f"No episode found for split={split}, offset={episode_offset}"
            ) from exc

        steps_dataset = episode["steps"]
        steps = list(steps_dataset.as_numpy_iterator())
        if not steps:
            raise ValueError("RLDS episode contains no steps")
        if frame_id < 0 or frame_id >= len(steps):
            raise IndexError(f"frame_id {frame_id} is outside [0, {len(steps) - 1}]")

        states = np.stack(
            [_as_numpy(step["observation"]["state"]) for step in steps], axis=0
        ).astype(np.float32)
        actions = np.stack([_as_numpy(step["action"]) for step in steps], axis=0).astype(
            np.float32
        )
        if states.shape[1:] != (STATE_DIM,):
            raise ValueError(f"Expected state shape (T, {STATE_DIM}), got {states.shape}")

        step = steps[frame_id]
        metadata = episode["episode_metadata"]
        episode_id = _decode_text(metadata["episode_id"])
        instruction = _decode_text(step["language_instruction"])
        front = _as_numpy(step["observation"]["image"]).astype(np.uint8, copy=False)
        wrist = _as_numpy(step["observation"]["wrist_image"]).astype(
            np.uint8, copy=False
        )

        return EpisodeSample(
            split=split,
            episode_offset=episode_offset,
            frame_id=frame_id,
            episode_id=episode_id,
            instruction=instruction,
            front_image=front,
            wrist_image=wrist,
            state=states[frame_id],
            action=actions[frame_id],
            action_chunk=build_action_chunk(actions, frame_id=frame_id),
            metadata={
                "episode_frame_count": len(steps),
                "suite": _decode_text(metadata["suite"]),
                "task": _decode_text(metadata["task"]),
                "source_episode_index": int(
                    _as_numpy(metadata["source_episode_index"]).reshape(()).item()
                ),
            },
        )
