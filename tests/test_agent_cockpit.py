from pathlib import Path

from src.agent_cockpit.evaluator import StateDeltaEvaluator
from src.agent_cockpit.models import AgentMode, AgentObservation
from src.agent_cockpit.orchestrator import AgentOrchestrator
from src.agent_cockpit.planner import RuleBasedPlanner
from src.agent_cockpit.safety import ActionSafetyValidator
from src.agent_cockpit.storage import TraceStore


def _orchestrator(root: Path) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=RuleBasedPlanner(),
        validator=ActionSafetyValidator(),
        evaluator=StateDeltaEvaluator(),
        trace_store=TraceStore(root),
    )


def test_proposal_mode_never_executes_and_persists_trace(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.trace_store.create_run("run_test", {"model_id": "mock"})
    result = orchestrator.run_step(
        run_id="run_test",
        goal="対象物を把持する",
        observation=AgentObservation(
            step_id=1,
            instruction="pick up the object",
            state=[0.0] * 8,
        ),
        mode=AgentMode.PROPOSE,
    )

    assert result["safety"]["passed"] is True
    assert result["execution"] == {"executed": False, "reason": "proposal_only"}
    assert (tmp_path / "run_test" / "steps" / "step_0001" / "proposal.json").exists()
    assert (tmp_path / "run_test" / "timeline.jsonl").exists()


def test_approval_mode_requires_approval(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.trace_store.create_run("run_approval", {})
    result = orchestrator.run_step(
        run_id="run_approval",
        goal="対象物へ接近する",
        observation=AgentObservation(
            step_id=0,
            instruction="pick up the object",
            state=[0.0] * 8,
        ),
        mode=AgentMode.APPROVAL,
        approved=False,
    )
    assert result["execution"]["reason"] == "human_approval_required"


def test_safety_rejects_oversized_action() -> None:
    proposal = RuleBasedPlanner().propose(
        AgentObservation(step_id=0, instruction="observe", state=[0.0] * 8),
        "observe",
    )
    unsafe = proposal.selected.__class__(
        action_id="unsafe",
        label="unsafe",
        score=1.0,
        action=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        expected_result="none",
        source="test",
    )
    result = ActionSafetyValidator().validate(unsafe)
    assert result.passed is False
    assert result.checks["motion_delta"] is False
