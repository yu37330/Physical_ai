from __future__ import annotations

import json
from pathlib import Path

from src.agent_cockpit.dataset_explorer import RLDSEpisodeReader
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
    # 3 episodes of 8 frames, split 2 train / 1 validation.
    assert report["frame_counts"] == {"train": 16, "validation": 8}
    assert report["regenerated_this_run"] is True

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

    reader = RLDSEpisodeReader(report["data_dir"])
    assert reader.available_splits() == ["train", "val"]
    sample = reader.read_sample(split="train", episode_offset=0, frame_id=0)
    assert sample.front_image.shape == (32, 32, 3)
    assert sample.wrist_image.shape == (32, 32, 3)
    assert sample.state.shape == (8,)
    assert sample.action.shape == (7,)
    assert sample.action_chunk.shape == (8, 7)
    assert sample.instruction
    assert sample.summary()["episode_frame_count"] == 8

    persisted_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert persisted_selection["counts"]["total"] == 3

    # Re-running reuses the prepared dataset, so _generate_examples never fires.
    # The frame counts must still be right: they feed dataset_manifest.json's
    # frame_count, which the submission report cites, and a rerun used to write 0.
    # Go through the CLI because the builder class registers globally and cannot
    # be defined twice in one process, which is also why this only ever showed up
    # on a second Colab run.
    import subprocess
    import sys

    rerun_report = tfds_root / "rerun_report.json"
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.data.convert_selected_lerobot_to_rlds",
            "--source-root", str(source_root),
            "--selection", str(selection_path),
            "--output-root", str(tfds_root),
            "--report", str(rerun_report),
            "--image-size", "32",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rebuilt = json.loads(rerun_report.read_text(encoding="utf-8"))
    assert rebuilt["regenerated_this_run"] is False
    assert rebuilt["frame_counts"] == {"train": 16, "validation": 8}
    assert rebuilt["episode_counts"] == {"train": 2, "validation": 1}
