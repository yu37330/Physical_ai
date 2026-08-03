"""OpenVLA PolicyとLIBERO環境を因果的な閉ループで評価する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .libero_executor import LiberoSimulationExecutor
from .models import ActionCandidate
from .safety import ActionSafetyValidator
from .storage import TraceStore


class ClosedLoopPolicy(Protocol):
    """提出用OfflinePolicyが満たす閉ループ評価契約。"""

    def reset(self, instruction: str, seed: int | None = None) -> None: ...

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class LiberoClosedLoopConfig:
    """閉ループ評価の停止・保存条件。"""

    max_steps: int = 300
    stop_on_safety_failure: bool = True
    persist_observation_images: bool = False
    max_abs_delta: float = 1.0
    max_gripper_abs: float = 1.0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_abs_delta <= 0:
            raise ValueError("max_abs_delta must be positive")
        if self.max_gripper_abs <= 0:
            raise ValueError("max_gripper_abs must be positive")


@dataclass
class LiberoClosedLoopRunner:
    """予測Actionを環境へ入力し、次観測で再推論する。"""

    executor: LiberoSimulationExecutor
    policy: ClosedLoopPolicy
    trace_store: TraceStore
    config: LiberoClosedLoopConfig

    def run(self, *, run_id: str) -> dict[str, Any]:
        run_dir = self.trace_store.create_run(
            run_id,
            {
                "mode": "libero_causal_closed_loop",
                "task": self.executor.config_dict(),
                "official_benchmark_init_state_used_for_evaluation": True,
                "official_benchmark_observations_persisted": bool(
                    self.config.persist_observation_images
                ),
                "official_benchmark_data_used_for_training": False,
            },
        )
        reset_result = self.executor.reset()
        self.policy.reset(
            instruction=self.executor.instruction,
            seed=self.executor.config.seed,
        )
        validator = ActionSafetyValidator(
            action_dim=7,
            max_abs_delta=self.config.max_abs_delta,
            max_gripper_abs=self.config.max_gripper_abs,
        )

        rows: list[dict[str, Any]] = []
        success = False
        stop_reason = "max_steps"
        try:
            for step_id in range(min(self.config.max_steps, self.executor.config.max_steps)):
                observation = self.executor.policy_observation()
                image_paths: dict[str, str] = {}
                if self.config.persist_observation_images:
                    front_path = self.trace_store.save_image_array(
                        run_id,
                        step_id,
                        "agentview_image.png",
                        observation["agentview_image"],
                    )
                    wrist_path = self.trace_store.save_image_array(
                        run_id,
                        step_id,
                        "eye_in_hand_image.png",
                        observation["robot0_eye_in_hand_image"],
                    )
                    image_paths = {
                        "agentview_image_path": str(front_path),
                        "eye_in_hand_image_path": str(wrist_path),
                    }

                started = perf_counter()
                action = np.asarray(
                    self.policy.get_action(observation),
                    dtype=np.float32,
                ).reshape(-1)
                latency_ms = (perf_counter() - started) * 1000.0
                candidate = ActionCandidate(
                    action_id=f"libero_policy_step_{step_id}",
                    label="OpenVLA closed-loop action",
                    score=1.0,
                    action=action.astype(float).tolist(),
                    expected_result="LIBERO環境を目標達成方向へ遷移させる",
                    source="openvla_oft_offline_policy",
                )
                safety = validator.validate(candidate)
                execution: dict[str, Any] = {
                    "executed": False,
                    "reason": "safety_validation_failed",
                }
                if safety.passed:
                    execution = {
                        "executed": True,
                        "result": self.executor.execute(action),
                    }

                step_payloads = {
                    "simulation_observation": {
                        "instruction": self.executor.instruction,
                        "step_id": step_id,
                        "keys": sorted(observation),
                        "shapes": {
                            key: list(np.asarray(value).shape)
                            for key, value in observation.items()
                        },
                        "observation_values_persisted": False,
                        **image_paths,
                    },
                    "policy_action": {
                        "action": action.astype(float).tolist(),
                        "latency_ms": latency_ms,
                    },
                    "safety": safety.to_dict(),
                    "execution": execution,
                }
                self.trace_store.save_step(
                    run_id=run_id,
                    step_id=step_id,
                    payloads=step_payloads,
                )

                execution_result = execution.get("result", {})
                row = {
                    "step_id": step_id,
                    "latency_ms": latency_ms,
                    "safety_passed": safety.passed,
                    "executed": bool(execution["executed"]),
                    "reward": execution_result.get("reward"),
                    "done": execution_result.get("done"),
                    "success": execution_result.get("success", False),
                }
                rows.append(row)

                if not safety.passed and self.config.stop_on_safety_failure:
                    stop_reason = "safety_validation_failed"
                    break
                if bool(execution_result.get("success")):
                    success = True
                    stop_reason = "task_success"
                    break
                if bool(execution_result.get("done")):
                    stop_reason = "environment_done_without_reward"
                    break
            else:
                stop_reason = "max_steps"
        finally:
            self.executor.close()

        latencies = [float(row["latency_ms"]) for row in rows]
        summary = {
            "run_id": run_id,
            "mode": "libero_causal_closed_loop",
            "causal_simulation": True,
            "replay_only": False,
            "success": success,
            "stop_reason": stop_reason,
            "executed_steps": len(rows),
            "mean_policy_call_latency_ms": mean(latencies) if latencies else None,
            "run_dir": str(run_dir),
            "task": reset_result,
            "config": asdict(self.config),
            "steps": rows,
            "official_benchmark_data_used_for_training": False,
        }
        summary_path = self.trace_store.save_run_artifact(
            run_id,
            "libero_closed_loop_summary",
            summary,
        )
        return {**summary, "summary_path": str(summary_path)}
