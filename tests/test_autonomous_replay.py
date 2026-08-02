from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agent_cockpit.autonomous_replay import (
    AutonomousReplayRunner,
    ReplayLoopConfig,
)
from src.agent_cockpit.dataset_explorer import ReplayEpisode
from src.agent_cockpit.evaluator import StateDeltaEvaluator
from src.agent_cockpit.models import (
    ActionCandidate,
    ActionProposal,
    AgentObservation,
)
from src.agent_cockpit.orchestrator import AgentOrchestrator
from src.agent_cockpit.safety import ActionSafetyValidator
from src.agent_cockpit.storage import TraceStore


class StaticPlanner:
    def __init__(self, action: list[float]) -> None:
        self.action = action

    def propose(self, observation: AgentObservation, goal: str) -> ActionProposal:
        candidate = ActionCandidate(
            action_id="static",
            label="Static test action",
            score=1.0,
            action=self.action,
            expected_result="test",
            source="test",
        )
        return ActionProposal(
            goal=goal,
            subgoal=observation.instruction,
            candidates=[candidate],
            selected_action_id="static",
            uncertainty=0.0,
        )


def _episode(action: list[float], frame_count: int = 5) -> ReplayEpisode:
    return ReplayEpisode(
        split="train",
        episode_offset=0,
        episode_id="episode_test",
        instruction="pick up the target",
        front_images=np.zeros((frame_count, 16, 16, 3), dtype=np.uint8),
        wrist_images=np.zeros((frame_count, 16, 16, 3), dtype=np.uint8),
        states=np.zeros((frame_count, 8), dtype=np.float32),
        actions=np.repeat(
            np.asarray(action, dtype=np.float32)[None, :],
            frame_count,
            axis=0,
        ),
        metadata={"suite": "test", "task": "test", "source_episode_index": 1},
    )


def _runner(
    tmp_path: Path,
    action: list[float],
    config: ReplayLoopConfig,
) -> tuple[AutonomousReplayRunner, TraceStore]:
    store = TraceStore(tmp_path)
    orchestrator = AgentOrchestrator(
        planner=StaticPlanner(action),
        validator=ActionSafetyValidator(),
        evaluator=StateDeltaEvaluator(),
        trace_store=store,
        executor=None,
    )
    return (
        AutonomousReplayRunner(
            orchestrator=orchestrator,
            trace_store=store,
            config=config,
        ),
        store,
    )


def test_replay_stops_after_repeated_actions(tmp_path: Path) -> None:
    action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = ReplayLoopConfig(
        max_steps=5,
        action_mae_threshold=0.01,
        max_consecutive_mismatches=3,
        max_repeated_actions=2,
    )
    runner, store = _runner(tmp_path, action, config)
    store.create_run("replay_test", {"mode": "test"})

    summary = runner.run(
        run_id="replay_test",
        goal="complete the task",
        episode=_episode(action),
    )

    assert summary["executed_steps"] == 2
    assert summary["stop_reason"] == "repeated_action_limit"
    assert summary["action_match_rate"] == 1.0
    assert Path(summary["summary_path"]).is_file()
    assert (tmp_path / "replay_test" / "steps" / "step_0000" / "front_image.png").is_file()
    assert (
        tmp_path
        / "replay_test"
        / "steps"
        / "step_0000"
        / "replay_evaluation.json"
    ).is_file()


def test_replay_stops_on_safety_failure(tmp_path: Path) -> None:
    unsafe_action = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = ReplayLoopConfig(
        max_steps=5,
        action_mae_threshold=1.0,
        max_consecutive_mismatches=3,
        max_repeated_actions=3,
        stop_on_safety_failure=True,
    )
    runner, store = _runner(tmp_path, unsafe_action, config)
    store.create_run("unsafe_replay", {"mode": "test"})

    summary = runner.run(
        run_id="unsafe_replay",
        goal="complete the task",
        episode=_episode(unsafe_action),
    )

    assert summary["executed_steps"] == 1
    assert summary["stop_reason"] == "safety_validation_failed"
    assert summary["safety_pass_rate"] == 0.0
