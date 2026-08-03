from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.agent_cockpit.dataset_explorer import EpisodeSample
from src.agent_cockpit.gpu_validation import (
    ColabGPUValidationRunner,
    GPUValidationConfig,
)
from src.agent_cockpit.libero_closed_loop import (
    LiberoClosedLoopConfig,
    LiberoClosedLoopRunner,
)
from src.agent_cockpit.libero_executor import (
    LiberoBindings,
    LiberoSimulationExecutor,
    LiberoTaskConfig,
)
from src.agent_cockpit.policy_adapter import InferenceResult
from src.agent_cockpit.storage import TraceStore


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "script",
    ["scripts/run_colab_gpu_validation.py", "scripts/run_libero_closed_loop.py"],
)
def test_runner_scripts_import_without_pythonpath(script: str) -> None:
    """Notebooks call these as `python scripts/...` with no PYTHONPATH set."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeCuda()


class _FakeReader:
    dataset_dir = Path("/fake/rlds")

    def available_splits(self) -> list[str]:
        return ["train", "val"]

    def read_sample(
        self,
        *,
        split: str,
        episode_offset: int,
        frame_id: int,
    ) -> EpisodeSample:
        del episode_offset
        if split != "val" or frame_id >= 2:
            raise IndexError(frame_id)
        actions = np.full((8, 7), frame_id * 0.01, dtype=np.float32)
        return EpisodeSample(
            split=split,
            episode_offset=0,
            frame_id=frame_id,
            episode_id="episode_000000",
            instruction="pick up the target object",
            front_image=np.zeros((16, 16, 3), dtype=np.uint8),
            wrist_image=np.zeros((16, 16, 3), dtype=np.uint8),
            state=np.zeros(8, dtype=np.float32),
            action=actions[0],
            action_chunk=actions,
            metadata={"episode_frame_count": 2},
        )


class _FakePolicyAdapter:
    checkpoint_dir = Path("/fake/checkpoint")

    def predict_rlds(self, **kwargs) -> InferenceResult:
        frame_value = float(np.asarray(kwargs["state"])[0])
        return InferenceResult(
            action_chunk=np.full((8, 7), frame_value, dtype=np.float32),
            latency_ms=12.5,
            checkpoint_dir=str(self.checkpoint_dir),
            source="fake",
        )


def test_gpu_validation_runner_collects_report(monkeypatch) -> None:
    from src.agent_cockpit import gpu_validation

    monkeypatch.setattr(
        gpu_validation,
        "collect_runtime_environment",
        lambda require_cuda: (
            {
                "cuda_available": False,
                "gpu": None,
                "packages": {},
            },
            _FakeTorch(),
        ),
    )
    report = ColabGPUValidationRunner(
        reader=_FakeReader(),
        policy=_FakePolicyAdapter(),
        config=GPUValidationConfig(
            split="val",
            num_frames=2,
            warmup_runs=0,
            require_cuda=False,
        ),
    ).run()

    assert report["status"] == "pass"
    assert report["checks"]["samples_validated"] == 2
    assert report["summary"]["mean_latency_ms"] == 12.5
    assert report["official_evaluation_data_used_for_training"] is False


def _observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


class _FakeEnv:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.steps = 0
        self.closed = False

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self):
        return _observation()

    def set_init_state(self, state):
        self.init_state = np.asarray(state)
        return _observation()

    def step(self, action):
        self.steps += 1
        success = self.steps >= 2
        return _observation(), float(success), bool(success), {"action": np.asarray(action)}

    def close(self) -> None:
        self.closed = True


class _FakeSuite:
    n_tasks = 1

    def get_task(self, task_id: int):
        assert task_id == 0
        return SimpleNamespace(
            language="pick up the target object",
            name="fake_task",
            problem_folder="fake",
            bddl_file="fake.bddl",
        )

    def get_task_init_states(self, task_id: int):
        assert task_id == 0
        return np.zeros((1, 4), dtype=np.float32)


class _FakeClosedLoopPolicy:
    def reset(self, instruction: str, seed: int | None = None) -> None:
        self.instruction = instruction
        self.seed = seed

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        assert "agentview_image" in observation
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def test_libero_closed_loop_stops_on_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHYSICAL_AI_ALLOW_MISSING_BDDL", "1")
    bindings = LiberoBindings(
        benchmark_dict={"libero_spatial": _FakeSuite},
        get_libero_path=lambda _: str(tmp_path),
        env_factory=_FakeEnv,
    )
    executor = LiberoSimulationExecutor(
        LiberoTaskConfig(
            task_suite_name="libero_spatial",
            task_id=0,
            settle_steps=0,
            max_steps=5,
        ),
        bindings=bindings,
    )
    summary = LiberoClosedLoopRunner(
        executor=executor,
        policy=_FakeClosedLoopPolicy(),
        trace_store=TraceStore(tmp_path / "traces"),
        config=LiberoClosedLoopConfig(max_steps=5),
    ).run(run_id="closed_loop_test")

    assert summary["success"] is True
    assert summary["stop_reason"] == "task_success"
    assert summary["executed_steps"] == 2
    assert Path(summary["summary_path"]).is_file()
    assert summary["official_benchmark_data_used_for_training"] is False
