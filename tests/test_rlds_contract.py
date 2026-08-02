from __future__ import annotations

import copy

import numpy as np

from src.data.rlds_contract import build_rlds_steps, validate_batch_contract, validate_episode_arrays
from src.data.update_dataset_manifest_after_rlds import update_manifest


def test_episode_and_batch_contract() -> None:
    states = np.zeros((10, 8), dtype=np.float32)
    actions = np.zeros((10, 7), dtype=np.float32)
    front = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    wrist = np.zeros((10, 32, 32, 3), dtype=np.uint8)

    result = validate_episode_arrays(
        states=states,
        actions=actions,
        front_images=front,
        wrist_images=wrist,
    )
    assert result.frame_count == 10

    steps = build_rlds_steps(
        states=states,
        actions=actions,
        front_images=front,
        wrist_images=wrist,
        instruction="pick up the bowl",
    )
    assert steps[0]["is_first"] is True
    assert steps[-1]["is_last"] is True
    assert steps[-1]["reward"] == np.float32(1.0)

    report = validate_batch_contract(
        {
            "pixel_values": np.zeros((1, 2, 3, 224, 224), dtype=np.float32),
            "input_ids": np.zeros((1, 64), dtype=np.int64),
            "labels": np.zeros((1, 64), dtype=np.int64),
            "actions": np.zeros((1, 8, 7), dtype=np.float32),
            "proprio": np.zeros((1, 8), dtype=np.float32),
        }
    )
    assert report["actions_shape"] == [1, 8, 7]
    assert report["proprio_shape"] == [1, 8]


def test_manifest_promoted_only_after_compatibility_pass() -> None:
    manifest = {
        "dataset_id": "parc_stage_a_balanced_v001",
        "structure": {"format": "lerobot_v2_1", "episode_count": 800},
        "transformations": [],
        "quality": {"checks": {}, "status": "metadata_selected_pending_payload_validation"},
    }
    conversion = {
        "dataset_name": "parc_libero_plus_selected",
        "version": "1.0.0",
        "selection_sha256": "a" * 64,
        "episode_counts": {"train": 640, "validation": 160},
        "contract": {
            "state_dim": 8,
            "action_dim": 7,
            "image_size": 256,
            "rotate_180_during_conversion": False,
        },
    }
    compatibility = {
        "status": "pass",
        "contract": {"action_chunk_length": 8, "action_dim": 7, "state_dim": 8},
        "checks": {
            "rlds_dataset_constructed": True,
            "rlds_batch_transform_passed": True,
            "collator_passed": True,
            "finite_action_and_proprio": True,
            "action_chunk_shape_passed": True,
        },
    }

    result = update_manifest(copy.deepcopy(manifest), conversion, compatibility)
    assert result["structure"]["format"] == "tfds_rlds"
    assert result["quality"]["status"] == "payload_validated"
    assert result["quality"]["checks"]["action_chunk_shape_passed"] is True
