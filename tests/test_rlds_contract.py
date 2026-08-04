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

    # PaddedCollatorForActionPrediction concatenates the wrist tensor into
    # pixel_values along dim=1, so a collated batch has one image tensor with the
    # channels of both cameras. Asserting a separate pixel_values_wrist key here
    # is what made the wrong contract look verified.
    report = validate_batch_contract(
        {
            "pixel_values": np.zeros((1, 6, 224, 224), dtype=np.float32),
            "input_ids": np.zeros((1, 64), dtype=np.int64),
            "labels": np.zeros((1, 64), dtype=np.int64),
            "actions": np.zeros((1, 8, 7), dtype=np.float32),
            "proprio": np.zeros((1, 8), dtype=np.float32),
        },
        single_image_channels=3,
    )
    assert report["actions_shape"] == [1, 8, 7]
    assert report["proprio_shape"] == [1, 8]
    assert report["pixel_values"]["shape"] == [1, 6, 224, 224]


def test_manifest_promoted_only_after_parity_and_compatibility_pass() -> None:
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
        "frame_counts": {"train": 6400, "validation": 1600},
        "tfds_splits": ["train", "val"],
        "contract": {
            "state_dim": 8,
            "action_dim": 7,
            "image_size": 256,
            "rotate_180_during_conversion": False,
        },
    }
    parity = {
        "status": "pass",
        "checks": {
            "state_raw_parity": True,
            "action_raw_parity": True,
            "front_orientation_preserved": True,
            "wrist_orientation_preserved": True,
            "front_pixel_parity": True,
            "wrist_pixel_parity": True,
        },
    }
    compatibility = {
        "status": "pass",
        "contract": {
            "action_chunk_length": 8,
            "action_dim": 7,
            "state_dim": 8,
            "front_tensor_key": "pixel_values",
            "wrist_tensor_key": "pixel_values_wrist",
        },
        "checks": {
            "rlds_dataset_constructed": True,
            "rlds_batch_transform_passed": True,
            "collator_passed": True,
            "front_and_wrist_tensors_present": True,
            "finite_action_and_proprio": True,
            "action_chunk_shape_passed": True,
        },
    }

    result = update_manifest(copy.deepcopy(manifest), conversion, parity, compatibility)
    assert result["structure"]["format"] == "tfds_rlds"
    assert result["structure"]["frame_count"] == 8000
    assert result["quality"]["status"] == "payload_validated"
    assert result["quality"]["checks"]["front_orientation_preserved"] is True
    assert result["quality"]["checks"]["action_chunk_shape_passed"] is True
