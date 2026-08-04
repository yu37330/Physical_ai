from __future__ import annotations

import numpy as np
import pytest

from src.data.rlds_contract import (
    ACTION_CHUNK_LENGTH,
    ACTION_DIM,
    STATE_DIM,
    validate_batch_contract,
    validate_transform_contract,
)

# Prismatic runs two vision backbones, so one image already carries several
# channels. The exact number does not matter here; the collated batch must carry
# twice it.
SINGLE_IMAGE_CHANNELS = 6


def _transform_sample(**overrides):
    sample = {
        "pixel_values": np.zeros((SINGLE_IMAGE_CHANNELS, 224, 224), dtype=np.float32),
        "pixel_values_wrist": np.zeros((SINGLE_IMAGE_CHANNELS, 224, 224), dtype=np.float32),
        "input_ids": np.zeros((32,), dtype=np.int64),
        "labels": np.zeros((32,), dtype=np.int64),
        "actions": np.zeros((ACTION_CHUNK_LENGTH, ACTION_DIM), dtype=np.float32),
    }
    sample.update(overrides)
    return sample


def _collated_batch(channels: int = 2 * SINGLE_IMAGE_CHANNELS, **overrides):
    batch = {
        "pixel_values": np.zeros((1, channels, 224, 224), dtype=np.float32),
        "input_ids": np.zeros((1, 32), dtype=np.int64),
        "labels": np.zeros((1, 32), dtype=np.int64),
        "actions": np.zeros((1, ACTION_CHUNK_LENGTH, ACTION_DIM), dtype=np.float32),
        "proprio": np.zeros((1, STATE_DIM), dtype=np.float32),
    }
    batch.update(overrides)
    return batch


def test_transform_output_carries_both_cameras() -> None:
    report = validate_transform_contract(_transform_sample())

    assert report["single_image_channels"] == SINGLE_IMAGE_CHANNELS


def test_missing_wrist_in_transform_output_is_rejected() -> None:
    sample = _transform_sample()
    del sample["pixel_values_wrist"]

    with pytest.raises(KeyError, match="pixel_values_wrist"):
        validate_transform_contract(sample)


def test_collated_batch_has_no_separate_wrist_key() -> None:
    """PaddedCollatorForActionPrediction concatenates wrist into pixel_values and
    drops the key, so requiring it here rejected a correct batch."""
    report = validate_batch_contract(
        _collated_batch(), single_image_channels=SINGLE_IMAGE_CHANNELS
    )

    assert report["actions_shape"] == [1, ACTION_CHUNK_LENGTH, ACTION_DIM]
    assert "pixel_values" in report


def test_a_dropped_wrist_stream_is_caught_by_the_channel_count() -> None:
    """With the key gone, halved channels are the only sign the wrist camera was
    not collated."""
    with pytest.raises(ValueError, match="Expected 12 pixel channels"):
        validate_batch_contract(
            _collated_batch(channels=SINGLE_IMAGE_CHANNELS),
            single_image_channels=SINGLE_IMAGE_CHANNELS,
        )


def test_action_chunk_shape_is_enforced() -> None:
    batch = _collated_batch(actions=np.zeros((1, 4, ACTION_DIM), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected action chunk"):
        validate_batch_contract(batch)


def test_non_finite_values_are_rejected() -> None:
    actions = np.zeros((1, ACTION_CHUNK_LENGTH, ACTION_DIM), dtype=np.float32)
    actions[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_batch_contract(_collated_batch(actions=actions))
