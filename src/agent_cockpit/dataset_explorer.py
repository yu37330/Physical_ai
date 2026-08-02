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


@dataclass(frozen=True)
class ReplayEpisode:
    """自律Replayでフレームを順番に観測するためのEpisode本体。"""

    split: str
    episode_offset: int
    episode_id: str
    instruction: str
    front_images: np.ndarray
    wrist_images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])

    def sample(self, frame_id: int) -> EpisodeSample:
        if frame_id < 0 or frame_id >= self.frame_count:
            raise IndexError(
                f"frame_id {frame_id} is outside [0, {self.frame_count - 1}]"
            )
        return EpisodeSample(
            split=self.split,
            episode_offset=self.episode_offset,
            frame_id=frame_id,
            episode_id=self.episode_id,
            instruction=self.instruction,
            front_image=self.front_images[frame_id],
            wrist_image=self.wrist_images[frame_id],
            state=self.states[frame_id],
            action=self.actions[frame_id],
            action_chunk=build_action_chunk(self.actions, frame_id=frame_id),
            metadata={
                "episode_frame_count": self.frame_count,
                **self.metadata,
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "episode_offset": self.episode_offset,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "frame_count": self.frame_count,
            "front_images_shape": list(self.front_images.shape),
            "wrist_images_shape": list(self.wrist_images.shape),
            "states_shape": list(self.states.shape),
            "actions_shape": list(self.actions.shape),
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

    def read_episode(self, *, split: str, episode_offset: int) -> ReplayEpisode:
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

        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            raise ValueError("RLDS episode contains no steps")

        states = np.stack(
            [_as_numpy(step["observation"]["state"]) for step in steps], axis=0
        ).astype(np.float32)
        actions = np.stack([_as_numpy(step["action"]) for step in steps], axis=0).astype(
            np.float32
        )
        front_images = np.stack(
            [_as_numpy(step["observation"]["image"]) for step in steps], axis=0
        ).astype(np.uint8)
        wrist_images = np.stack(
            [_as_numpy(step["observation"]["wrist_image"]) for step in steps], axis=0
        ).astype(np.uint8)

        if states.shape[1:] != (STATE_DIM,):
            raise ValueError(f"Expected state shape (T, {STATE_DIM}), got {states.shape}")
        if actions.shape[1:] != (ACTION_DIM,):
            raise ValueError(f"Expected action shape (T, {ACTION_DIM}), got {actions.shape}")
        stream_lengths = {
            states.shape[0],
            actions.shape[0],
            front_images.shape[0],
            wrist_images.shape[0],
        }
        if len(stream_lengths) != 1:
            raise ValueError("RLDS episode stream lengths do not match")

        metadata = episode["episode_metadata"]
        return ReplayEpisode(
            split=split,
            episode_offset=episode_offset,
            episode_id=_decode_text(metadata["episode_id"]),
            instruction=_decode_text(steps[0]["language_instruction"]),
            front_images=front_images,
            wrist_images=wrist_images,
            states=states,
            actions=actions,
            metadata={
                "suite": _decode_text(metadata["suite"]),
                "task": _decode_text(metadata["task"]),
                "source_episode_index": int(
                    _as_numpy(metadata["source_episode_index"]).reshape(()).item()
                ),
            },
        )

    def read_sample(
        self,
        *,
        split: str,
        episode_offset: int,
        frame_id: int,
    ) -> EpisodeSample:
        return self.read_episode(
            split=split,
            episode_offset=episode_offset,
        ).sample(frame_id)
