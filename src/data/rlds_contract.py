from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DATASET_NAME = "parc_libero_plus_selected"
DATASET_VERSION = "1.0.0"
STATE_DIM = 8
ACTION_DIM = 7
ACTION_CHUNK_LENGTH = 8
FRONT_KEY = "image"
WRIST_KEY = "wrist_image"
STATE_KEY = "state"


@dataclass(frozen=True)
class EpisodeContractResult:
    frame_count: int
    state_shape: tuple[int, int]
    action_shape: tuple[int, int]
    front_shape: tuple[int, int, int, int]
    wrist_shape: tuple[int, int, int, int]


def _as_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def validate_episode_arrays(
    *,
    states: Any,
    actions: Any,
    front_images: Any,
    wrist_images: Any,
) -> EpisodeContractResult:
    state_array = _as_array(states, name="states")
    action_array = _as_array(actions, name="actions")
    front_array = _as_array(front_images, name="front_images")
    wrist_array = _as_array(wrist_images, name="wrist_images")

    if state_array.ndim != 2 or state_array.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states (T, {STATE_DIM}), got {state_array.shape}")
    if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions (T, {ACTION_DIM}), got {action_array.shape}")
    for name, array in (("front_images", front_array), ("wrist_images", wrist_array)):
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"Expected {name} (T, H, W, 3), got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"Expected {name} uint8, got {array.dtype}")

    frame_count = state_array.shape[0]
    lengths = {
        "states": state_array.shape[0],
        "actions": action_array.shape[0],
        "front_images": front_array.shape[0],
        "wrist_images": wrist_array.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode stream lengths do not match: {lengths}")
    if not np.isfinite(state_array).all():
        raise ValueError("states contain NaN or Inf")
    if not np.isfinite(action_array).all():
        raise ValueError("actions contain NaN or Inf")

    return EpisodeContractResult(
        frame_count=frame_count,
        state_shape=tuple(state_array.shape),
        action_shape=tuple(action_array.shape),
        front_shape=tuple(front_array.shape),
        wrist_shape=tuple(wrist_array.shape),
    )


def build_rlds_steps(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    front_images: np.ndarray,
    wrist_images: np.ndarray,
    instruction: str,
) -> list[dict[str, Any]]:
    result = validate_episode_arrays(
        states=states,
        actions=actions,
        front_images=front_images,
        wrist_images=wrist_images,
    )
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")

    steps: list[dict[str, Any]] = []
    for index in range(result.frame_count):
        is_last = index == result.frame_count - 1
        steps.append(
            {
                "observation": {
                    FRONT_KEY: front_images[index],
                    WRIST_KEY: wrist_images[index],
                    STATE_KEY: states[index].astype(np.float32, copy=False),
                },
                "action": actions[index].astype(np.float32, copy=False),
                "discount": np.float32(0.0 if is_last else 1.0),
                "reward": np.float32(1.0 if is_last else 0.0),
                "is_first": index == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "language_instruction": instruction,
            }
        )
    return steps


def validate_batch_contract(batch: Mapping[str, Any]) -> dict[str, Any]:
    required = {"pixel_values", "input_ids", "labels", "actions", "proprio"}
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"RLDSBatchTransform output is missing: {missing}")

    actions = np.asarray(batch["actions"])
    proprio = np.asarray(batch["proprio"])
    if actions.shape[-2:] != (ACTION_CHUNK_LENGTH, ACTION_DIM):
        raise ValueError(
            f"Expected action chunk (..., {ACTION_CHUNK_LENGTH}, {ACTION_DIM}), got {actions.shape}"
        )
    if proprio.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected proprio (..., {STATE_DIM}), got {proprio.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
        raise ValueError("Transformed actions/proprio contain NaN or Inf")

    pixel_values = np.asarray(batch["pixel_values"])
    if pixel_values.ndim not in (4, 5):
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

    return {
        "pixel_values_shape": list(pixel_values.shape),
        "input_ids_shape": list(np.asarray(batch["input_ids"]).shape),
        "labels_shape": list(np.asarray(batch["labels"]).shape),
        "actions_shape": list(actions.shape),
        "proprio_shape": list(proprio.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "proprio_min": float(proprio.min()),
        "proprio_max": float(proprio.max()),
    }
