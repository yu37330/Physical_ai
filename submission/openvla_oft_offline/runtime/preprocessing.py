from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PolicyInput:
    full_image: Image.Image
    wrist_image: Image.Image
    proprio: np.ndarray
    instruction: str


def quaternion_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quaternion.shape}")

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    x, y, z, w = quaternion / norm
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = np.asarray([x, y, z], dtype=np.float64) / sin_half
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return (axis * angle).astype(np.float32)


def center_crop_resize(image: np.ndarray, crop_area: float = 0.9, size: int = 224) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    pil = Image.fromarray(array, mode="RGB")
    width, height = pil.size
    scale = math.sqrt(crop_area)
    crop_width = max(1, int(round(width * scale)))
    crop_height = max(1, int(round(height * scale)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    pil = pil.crop((left, top, left + crop_width, top + crop_height))
    return pil.resize((size, size), resample=Image.Resampling.LANCZOS)


def build_policy_input(observation: dict[str, np.ndarray], instruction: str) -> PolicyInput:
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [name for name in required if name not in observation]
    if missing:
        raise KeyError(f"Missing observation fields: {missing}")

    eef_pos = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3)
    axis_angle = quaternion_xyzw_to_axis_angle(
        np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(4)
    )
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
    proprio = np.concatenate((eef_pos, axis_angle, gripper)).astype(np.float32)

    return PolicyInput(
        full_image=center_crop_resize(observation["agentview_image"]),
        wrist_image=center_crop_resize(observation["robot0_eye_in_hand_image"]),
        proprio=proprio,
        instruction=str(instruction),
    )
