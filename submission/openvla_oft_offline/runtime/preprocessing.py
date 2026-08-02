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
    """Match the official OpenVLA-OFT LIBERO quat2axisangle implementation."""
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (
        quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator
    ).astype(np.float32)


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    """Rotate the raw LIBERO camera image 180 degrees to match training preprocessing."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {array.shape}")
    return np.ascontiguousarray(array[::-1, ::-1])


def resize_then_center_crop(image: np.ndarray, size: int = 224, crop_area: float = 0.9) -> Image.Image:
    """Approximate official TF Lanczos resize + 90% center crop with PIL."""
    array = rotate_libero_image(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    pil = Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    scale = math.sqrt(crop_area)
    crop_width = max(1, int(round(size * scale)))
    crop_height = max(1, int(round(size * scale)))
    left = (size - crop_width) // 2
    top = (size - crop_height) // 2
    return pil.crop((left, top, left + crop_width, top + crop_height)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


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
    axis_angle = quaternion_xyzw_to_axis_angle(observation["robot0_eef_quat"])
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
    proprio = np.concatenate((eef_pos, axis_angle, gripper)).astype(np.float32)

    return PolicyInput(
        full_image=resize_then_center_crop(observation["agentview_image"]),
        wrist_image=resize_then_center_crop(observation["robot0_eye_in_hand_image"]),
        proprio=proprio,
        instruction=str(instruction),
    )
