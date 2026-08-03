"""RLDS軌跡上でObserve→Plan→Evaluate→Replanを反復する。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np

from .dataset_explorer import ReplayEpisode
from .models import AgentMode, AgentObservation
from .orchestrator import AgentOrchestrator
from .policy_adapter import compare_action_chunks
from .storage import TraceStore


@dataclass(frozen=True)
class ReplayLoopConfig:
    """オフライン自律Replayの停止条件。"""

    max_steps: int = 5
    action_mae_threshold: float = 0.25
    max_consecutive_mismatches: int = 3
    max_repeated_actions: int = 3
    stop_on_safety_failure: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.action_mae_threshold < 0:
            raise ValueError("action_mae_threshold must be non-negative")
        if self.max_consecutive_mismatches <= 0:
            raise ValueError("max_consecutive_mismatches must be positive")
        if self.max_repeated_actions <= 0:
            raise ValueError("max_repeated_actions must be positive")


@dataclass
class AutonomousReplayRunner:
    """記録済みRLDS観測を順に与え、各Frameで次Actionを再計画する。"""

    orchestrator: AgentOrchestrator
    trace_store: TraceStore
    config: ReplayLoopConfig

    def run(
        self,
        *,
        run_id: str,
        goal: str,
        episode: ReplayEpisode,
        start_frame: int = 0,
    ) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("goal must not be empty")
        if start_frame < 0 or start_frame >= episode.frame_count:
            raise IndexError(
                f"start_frame {start_frame} is outside [0, {episode.frame_count - 1}]"
            )

        rows: list[dict[str, Any]] = []
        previous_action: np.ndarray | None = None
        repeated_actions = 0
        consecutive_mismatches = 0
        stop_reason = "max_steps"
        last_frame = min(
            episode.frame_count,
            start_frame + self.config.max_steps,
        )

        for loop_step, frame_id in enumerate(range(start_frame, last_frame)):
            sample = episode.sample(frame_id)
            front_path = self.trace_store.save_image_array(
                run_id,
                loop_step,
                "front_image.png",
                sample.front_image,
            )
            wrist_path = self.trace_store.save_image_array(
                run_id,
                loop_step,
                "wrist_image.png",
                sample.wrist_image,
            )
            observation = AgentObservation(
                step_id=loop_step,
                instruction=sample.instruction,
                state=sample.state.astype(float).tolist(),
                image_path=str(front_path),
                metadata={
                    "wrist_image_path": str(wrist_path),
                    "source": "rlds_autonomous_replay",
                    "episode_id": sample.episode_id,
                    "frame_id": frame_id,
                },
            )
            after_state = (
                episode.states[frame_id + 1].astype(float).tolist()
                if frame_id + 1 < episode.frame_count
                else None
            )
            result = self.orchestrator.run_step(
                run_id=run_id,
                goal=goal,
                observation=observation,
                mode=AgentMode.PROPOSE,
                approved=False,
                after_state=after_state,
            )

            selected_action = np.asarray(
                result["proposal"]["selected"]["action"],
                dtype=np.float32,
            )
            target_action = sample.action.astype(np.float32)
            action_mae = float(np.mean(np.abs(selected_action - target_action)))
            action_rmse = float(
                np.sqrt(np.mean(np.square(selected_action - target_action)))
            )
            action_matches = action_mae <= self.config.action_mae_threshold
            consecutive_mismatches = (
                0 if action_matches else consecutive_mismatches + 1
            )

            if previous_action is not None and np.allclose(
                selected_action,
                previous_action,
                atol=1e-6,
                rtol=0.0,
            ):
                repeated_actions += 1
            else:
                repeated_actions = 1
            previous_action = selected_action.copy()

            proposal_metadata = result["proposal"].get("metadata", {})
            predicted_chunk = proposal_metadata.get("action_chunk")
            chunk_metrics = (
                compare_action_chunks(predicted_chunk, sample.action_chunk)
                if predicted_chunk is not None
                else {"status": "policy_chunk_not_available"}
            )
            replay_evaluation = {
                "episode_id": sample.episode_id,
                "episode_offset": sample.episode_offset,
                "frame_id": frame_id,
                "selected_action": selected_action.astype(float).tolist(),
                "target_action": target_action.astype(float).tolist(),
                "action_mae": action_mae,
                "action_rmse": action_rmse,
                "action_matches_threshold": action_matches,
                "action_mae_threshold": self.config.action_mae_threshold,
                "consecutive_mismatches": consecutive_mismatches,
                "repeated_action_count": repeated_actions,
                "chunk_metrics": chunk_metrics,
                "replay_only": True,
                "causal_simulation": False,
                "note": (
                    "次観測は記録済みRLDS軌跡から取得するため、予測Actionによる"
                    "環境遷移を再現するシミュレーションではありません。"
                ),
            }
            self.trace_store.save_step_artifact(
                run_id,
                loop_step,
                "replay_evaluation",
                replay_evaluation,
            )

            step_stop_reason: str | None = None
            safety_passed = bool(result["safety"]["passed"])
            if self.config.stop_on_safety_failure and not safety_passed:
                step_stop_reason = "safety_validation_failed"
            elif consecutive_mismatches >= self.config.max_consecutive_mismatches:
                step_stop_reason = "consecutive_action_mismatches"
            elif repeated_actions >= self.config.max_repeated_actions:
                step_stop_reason = "repeated_action_limit"
            elif frame_id + 1 >= episode.frame_count:
                step_stop_reason = "episode_end"

            rows.append(
                {
                    "loop_step": loop_step,
                    "frame_id": frame_id,
                    "action_mae": action_mae,
                    "action_rmse": action_rmse,
                    "action_match": action_matches,
                    "safety_passed": safety_passed,
                    "repeated_action_count": repeated_actions,
                    "consecutive_mismatches": consecutive_mismatches,
                    "latency_ms": proposal_metadata.get("latency_ms"),
                    "stop_reason": step_stop_reason or "",
                }
            )
            if step_stop_reason:
                stop_reason = step_stop_reason
                break
        else:
            if last_frame >= episode.frame_count:
                stop_reason = "episode_end"

        action_maes = [float(row["action_mae"]) for row in rows]
        summary = {
            "run_id": run_id,
            "mode": "offline_autonomous_replay",
            "episode": episode.summary(),
            "goal": goal.strip(),
            "start_frame": start_frame,
            "executed_steps": len(rows),
            "stop_reason": stop_reason,
            "mean_action_mae": mean(action_maes) if action_maes else None,
            "action_match_rate": (
                sum(bool(row["action_match"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "safety_pass_rate": (
                sum(bool(row["safety_passed"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "replay_only": True,
            "causal_simulation": False,
            "config": {
                "max_steps": self.config.max_steps,
                "action_mae_threshold": self.config.action_mae_threshold,
                "max_consecutive_mismatches": self.config.max_consecutive_mismatches,
                "max_repeated_actions": self.config.max_repeated_actions,
                "stop_on_safety_failure": self.config.stop_on_safety_failure,
            },
            "steps": rows,
        }
        summary_path = self.trace_store.save_run_artifact(
            run_id,
            "autonomous_replay_summary",
            summary,
        )
        return {
            **summary,
            "summary_path": str(summary_path),
        }
