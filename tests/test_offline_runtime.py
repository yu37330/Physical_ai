from __future__ import annotations

import numpy as np

from submission.openvla_oft_offline.runtime.action_postprocess import (
    ActionChunkBuffer,
    finalize_action,
)
from submission.openvla_oft_offline.runtime.preprocessing import (
    build_policy_input,
    quaternion_xyzw_to_axis_angle,
)


def test_identity_quaternion_to_zero_axis_angle() -> None:
    result = quaternion_xyzw_to_axis_angle(np.asarray([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-7)


def test_build_policy_input_shapes() -> None:
    observation = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    result = build_policy_input(observation, "move the bowl")
    assert result.full_image.size == (224, 224)
    assert result.wrist_image.size == (224, 224)
    assert result.proprio.shape == (8,)


def test_action_chunk_buffer() -> None:
    buffer = ActionChunkBuffer()
    chunk = np.arange(56, dtype=np.float32).reshape(8, 7)
    buffer.load(chunk)
    np.testing.assert_array_equal(buffer.pop(), chunk[0])


def test_gripper_postprocess_matches_libero_convention() -> None:
    action = np.zeros(7, dtype=np.float32)
    action[-1] = 1.0
    result = finalize_action(action)
    assert result[-1] == -1.0
