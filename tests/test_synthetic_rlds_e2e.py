from __future__ import annotations

import json
from pathlib import Path

from src.data.convert_selected_lerobot_to_rlds import _build_tfds
from src.data.generate_synthetic_lerobot_fixture import generate_fixture
from src.data.validate_rlds_source_parity import validate_episode


def test_synthetic_lerobot_to_rlds_and_source_parity(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    selection_path = tmp_path / "selection.json"
    tfds_root = tmp_path / "tfds"

    selection = generate_fixture(
        source_root,
        selection_path,
        frame_count=8,
        image_size=32,
        fps=20,
    )
    report = _build_tfds(
        source_root=source_root,
        selection=selection,
        output_root=tfds_root,
        front_video_key="observation.images.front",
        wrist_video_key="observation.images.wrist",
        state_column="observation.state",
        action_column="action",
        image_size=32,
        rotate_180=False,
    )

    assert report["episode_counts"] == {"train": 2, "validation": 1}
    assert report["tfds_splits"] == ["train", "val"]
    assert report["contract"]["state_dim"] == 8
    assert report["contract"]["action_dim"] == 7

    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(report["data_dir"])
    split_expectations = {"train": 2, "val": 1}
    for split, expected_count in split_expectations.items():
        episodes = list(builder.as_dataset(split=split, shuffle_files=False).take(expected_count))
        assert len(episodes) == expected_count
        for episode in episodes:
            parity = validate_episode(
                source_root=source_root,
                episode=episode,
                front_video_key="observation.images.front",
                wrist_video_key="observation.images.wrist",
                state_column="observation.state",
                action_column="action",
                image_size=32,
            )
            assert parity["passed"] is True
            assert parity["state_max_abs_error"] <= 1e-6
            assert parity["action_max_abs_error"] <= 1e-6
            assert parity["front"]["orientation_preserved"] is True
            assert parity["wrist"]["orientation_preserved"] is True

    persisted_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert persisted_selection["counts"]["total"] == 3
