from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.agent_cockpit.dataset_explorer import (
    build_action_chunk,
    build_dataset_overview,
    build_episode_table,
)
from src.agent_cockpit.models import AgentObservation
from src.agent_cockpit.policy_adapter import (
    OpenVLAPolicyAdapter,
    OpenVLAPolicyPlanner,
    compare_action_chunks,
)
from src.agent_cockpit.storage import TraceStore


class FakeChunkRuntime:
    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        self.last_policy_input = None

    def predict_chunk(self, policy_input):
        self.last_policy_input = policy_input
        return self.chunk


def test_build_action_chunk_pads_episode_tail() -> None:
    actions = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)

    chunk = build_action_chunk(actions, frame_id=3, chunk_length=4)

    assert chunk.shape == (4, 7)
    np.testing.assert_allclose(chunk[0], actions[3])
    np.testing.assert_allclose(chunk[1], actions[4])
    np.testing.assert_allclose(chunk[2], actions[4])
    np.testing.assert_allclose(chunk[3], actions[4])


def test_dataset_overview_and_episode_table(tmp_path: Path) -> None:
    manifest = {
        "dataset_id": "ds_test",
        "structure": {
            "format": "tfds_rlds",
            "episode_count": 2,
            "frame_count": 10,
        },
        "quality": {"status": "payload_validated", "checks": {"rlds": True}},
        "allowed_for_training": True,
        "contains_official_evaluation_data": False,
    }
    conversion = {
        "dataset_name": "parc_libero_plus_selected",
        "version": "1.0.0",
        "data_dir": "/tmp/rlds",
        "episode_counts": {"train": 1, "validation": 1},
        "frame_counts": {"train": 5, "validation": 5},
        "contract": {"state_dim": 8, "action_dim": 7},
    }
    compatibility = {
        "status": "pass",
        "checks": {"action_chunk_shape_passed": True},
        "contract": {"actions_shape": [1, 8, 7]},
    }
    selection = {
        "episodes": [
            {
                "episode_index": 12,
                "split": "train",
                "suite": "spatial",
                "instruction": "pick up the bowl",
                "length": 5,
            }
        ]
    }
    paths = {}
    for name, payload in (
        ("manifest", manifest),
        ("conversion", conversion),
        ("compatibility", compatibility),
        ("selection", selection),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    overview = build_dataset_overview(
        manifest_path=paths["manifest"],
        conversion_report_path=paths["conversion"],
        compatibility_report_path=paths["compatibility"],
    )
    rows = build_episode_table(paths["selection"])

    assert overview["dataset_id"] == "ds_test"
    assert overview["openvla_compatibility"]["status"] == "pass"
    assert rows == [[12, "train", "spatial", "pick up the bowl", 5]]


def test_openvla_adapter_preserves_rlds_orientation_and_compares_chunks() -> None:
    predicted = np.full((8, 7), 0.25, dtype=np.float32)
    runtime = FakeChunkRuntime(predicted)
    adapter = OpenVLAPolicyAdapter("unused-in-test", runtime=runtime)
    front = np.zeros((32, 32, 3), dtype=np.uint8)
    front[0, 0] = [255, 0, 0]
    wrist = np.zeros((32, 32, 3), dtype=np.uint8)

    result = adapter.predict_rlds(
        front_image=front,
        wrist_image=wrist,
        state=np.zeros(8, dtype=np.float32),
        instruction="pick up the target",
    )
    metrics = compare_action_chunks(result.action_chunk, np.zeros((8, 7)))

    assert result.action_chunk.shape == (8, 7)
    assert runtime.last_policy_input.proprio.shape == (8,)
    assert metrics["mae"] == 0.25
    assert metrics["rmse"] == 0.25


def test_openvla_planner_keeps_full_chunk_in_metadata(tmp_path: Path) -> None:
    chunk = np.arange(8 * 7, dtype=np.float32).reshape(8, 7) / 100.0
    adapter = OpenVLAPolicyAdapter("unused-in-test", runtime=FakeChunkRuntime(chunk))
    planner = OpenVLAPolicyPlanner(adapter)
    front_path = tmp_path / "front.png"
    wrist_path = tmp_path / "wrist.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(front_path)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(wrist_path)
    observation = AgentObservation(
        step_id=0,
        instruction="pick up the target",
        state=[0.0] * 8,
        image_path=str(front_path),
        metadata={"wrist_image_path": str(wrist_path)},
    )

    proposal = planner.propose(observation, "complete the task")

    assert proposal.selected.action == chunk[0].astype(float).tolist()
    assert proposal.metadata["action_chunk_shape"] == [8, 7]
    assert proposal.metadata["confidence_available"] is False


def test_trace_store_persists_observation_images(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.create_run("run_test", {"model_id": "dummy"})
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    saved = store.save_image_array("run_test", 0, "front_image.png", image)

    assert saved.is_file()
    assert saved.parent.name == "step_0000"
