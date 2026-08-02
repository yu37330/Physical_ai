"""LIBEROの因果的な閉ループ評価をAgent Cockpitへ接続する。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class LiberoBindings:
    """LIBERO依存を遅延Import・テスト差し替えするための束。"""

    benchmark_dict: Mapping[str, Callable[[], Any]]
    get_libero_path: Callable[[str], str]
    env_factory: Callable[..., Any]


def load_libero_bindings() -> LiberoBindings:
    """インストール済み公式LIBEROから必要なAPIだけを読み込む。"""

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO is not installed. Run training/openvla_oft_a100/"
            "scripts/bootstrap_libero_colab.sh first."
        ) from exc
    return LiberoBindings(
        benchmark_dict=benchmark.get_benchmark_dict(),
        get_libero_path=get_libero_path,
        env_factory=OffScreenRenderEnv,
    )


@dataclass(frozen=True)
class LiberoTaskConfig:
    """1回のLIBERO閉ループ評価条件。"""

    task_suite_name: str
    task_id: int
    init_state_id: int = 0
    seed: int = 7
    resolution: int = 256
    settle_steps: int = 10
    max_steps: int = 300

    def __post_init__(self) -> None:
        if not self.task_suite_name.strip():
            raise ValueError("task_suite_name must not be empty")
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.init_state_id < 0:
            raise ValueError("init_state_id must be non-negative")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _observation_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        summary[str(key)] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
        }
    return summary


class LiberoSimulationExecutor:
    """公式OffScreenRenderEnvを1 Actionずつ進める安全なExecutor。"""

    REQUIRED_POLICY_KEYS = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_joint_pos",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )

    def __init__(
        self,
        config: LiberoTaskConfig,
        *,
        bindings: LiberoBindings | None = None,
    ) -> None:
        self.config = config
        self.bindings = bindings or load_libero_bindings()
        self._env: Any | None = None
        self._task_suite: Any | None = None
        self._task: Any | None = None
        self._observation: dict[str, Any] | None = None
        self._instruction = ""
        self._step_count = 0

    @property
    def instruction(self) -> str:
        if not self._instruction:
            raise RuntimeError("LIBERO environment has not been reset")
        return self._instruction

    @property
    def step_count(self) -> int:
        return self._step_count

    @staticmethod
    def dummy_action() -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0
        return action

    def reset(self) -> dict[str, Any]:
        suite_name = self.config.task_suite_name
        if suite_name not in self.bindings.benchmark_dict:
            raise KeyError(
                f"Unknown LIBERO task suite {suite_name!r}; "
                f"available={sorted(self.bindings.benchmark_dict)}"
            )
        self.close()
        self._task_suite = self.bindings.benchmark_dict[suite_name]()
        if self.config.task_id >= int(self._task_suite.n_tasks):
            raise IndexError(
                f"task_id {self.config.task_id} is outside "
                f"[0, {int(self._task_suite.n_tasks) - 1}]"
            )
        self._task = self._task_suite.get_task(self.config.task_id)
        self._instruction = str(self._task.language)
        bddl_file = Path(self.bindings.get_libero_path("bddl_files")) / str(
            self._task.problem_folder
        ) / str(self._task.bddl_file)
        if not bddl_file.is_file() and os.environ.get("PHYSICAL_AI_ALLOW_MISSING_BDDL") != "1":
            raise FileNotFoundError(f"LIBERO BDDL file not found: {bddl_file}")

        self._env = self.bindings.env_factory(
            bddl_file_name=str(bddl_file),
            camera_heights=self.config.resolution,
            camera_widths=self.config.resolution,
        )
        if hasattr(self._env, "seed"):
            self._env.seed(self.config.seed)
        observation = self._env.reset()
        init_states = self._task_suite.get_task_init_states(self.config.task_id)
        if self.config.init_state_id >= len(init_states):
            raise IndexError(
                f"init_state_id {self.config.init_state_id} is outside "
                f"[0, {len(init_states) - 1}]"
            )
        set_result = self._env.set_init_state(init_states[self.config.init_state_id])
        if isinstance(set_result, Mapping):
            observation = set_result

        for _ in range(self.config.settle_steps):
            observation, _, _, _ = self._env.step(self.dummy_action())
        if not isinstance(observation, Mapping):
            raise TypeError("LIBERO environment reset did not return an observation mapping")
        self._observation = dict(observation)
        self._step_count = 0
        self._validate_policy_observation(self._observation)
        return {
            "task_suite_name": suite_name,
            "task_id": self.config.task_id,
            "task_name": str(self._task.name),
            "instruction": self._instruction,
            "init_state_id": self.config.init_state_id,
            "seed": self.config.seed,
            "resolution": self.config.resolution,
            "observation": _observation_summary(self._observation),
            "official_benchmark_init_state_used_for_evaluation": True,
            "official_benchmark_init_state_persisted": False,
            "official_benchmark_data_used_for_training": False,
        }

    def _validate_policy_observation(self, observation: Mapping[str, Any]) -> None:
        missing = [key for key in self.REQUIRED_POLICY_KEYS if key not in observation]
        if missing:
            raise KeyError(f"LIBERO observation is missing policy fields: {missing}")
        for image_key in ("agentview_image", "robot0_eye_in_hand_image"):
            image = np.asarray(observation[image_key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected {image_key} (H, W, 3), got {image.shape}")

    def policy_observation(self) -> dict[str, np.ndarray]:
        if self._observation is None:
            raise RuntimeError("LIBERO environment has not been reset")
        self._validate_policy_observation(self._observation)
        return {
            key: np.asarray(self._observation[key]).copy()
            for key in self.REQUIRED_POLICY_KEYS
        }

    def execute(self, action: list[float] | np.ndarray) -> dict[str, Any]:
        if self._env is None or self._observation is None:
            raise RuntimeError("LIBERO environment has not been reset")
        if self._step_count >= self.config.max_steps:
            raise RuntimeError("LIBERO max_steps has already been reached")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (7,):
            raise ValueError(f"Expected LIBERO action (7,), got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("LIBERO action contains NaN or Inf")

        observation, reward, done, info = self._env.step(action_array)
        if not isinstance(observation, Mapping):
            raise TypeError("LIBERO env.step did not return an observation mapping")
        self._observation = dict(observation)
        self._validate_policy_observation(self._observation)
        self._step_count += 1
        reward_value = float(np.asarray(reward).reshape(()).item())
        done_value = bool(np.asarray(done).reshape(()).item())
        return {
            "step_count": self._step_count,
            "reward": reward_value,
            "done": done_value,
            "success": reward_value > 0.0,
            "info": _jsonable(info),
            "observation": _observation_summary(self._observation),
        }

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
        self._observation = None
        self._step_count = 0

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
