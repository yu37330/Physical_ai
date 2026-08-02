"""Observe→Plan→Validate→Evaluate→Persistを統括する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .evaluator import StateDeltaEvaluator
from .models import AgentMode, AgentObservation
from .planner import RuleBasedPlanner
from .safety import ActionSafetyValidator
from .storage import TraceStore


class Executor(Protocol):
    """Simulationまたは実機Executorの最小契約。"""

    def execute(self, action: list[float]) -> dict[str, object]: ...


@dataclass
class AgentOrchestrator:
    """1ステップ分のAgent判断を監査可能な形で生成する。"""

    planner: RuleBasedPlanner
    validator: ActionSafetyValidator
    evaluator: StateDeltaEvaluator
    trace_store: TraceStore
    executor: Executor | None = None

    def run_step(
        self,
        *,
        run_id: str,
        goal: str,
        observation: AgentObservation,
        mode: AgentMode = AgentMode.PROPOSE,
        approved: bool = False,
        after_state: list[float] | None = None,
    ) -> dict[str, object]:
        proposal = self.planner.propose(observation, goal)
        selected = proposal.selected
        safety = self.validator.validate(selected)

        can_execute = self._can_execute(
            mode=mode,
            approved=approved,
            safety_passed=safety.passed,
        )
        execution: dict[str, object] = {
            "executed": False,
            "reason": "proposal_only",
        }
        if can_execute:
            if self.executor is None:
                execution = {
                    "executed": False,
                    "reason": "executor_not_configured",
                }
            else:
                execution = {
                    "executed": True,
                    "result": self.executor.execute(selected.action),
                }
        elif not safety.passed:
            execution = {
                "executed": False,
                "reason": "safety_validation_failed",
            }
        elif mode == AgentMode.APPROVAL and not approved:
            execution = {
                "executed": False,
                "reason": "human_approval_required",
            }

        evaluation = self.evaluator.evaluate(observation.state, after_state)
        trace_payloads = {
            "observation": observation.to_dict(),
            "proposal": proposal.to_dict(),
            "safety": safety.to_dict(),
            "execution": execution,
            "evaluation": evaluation.to_dict(),
        }
        step_dir = self.trace_store.save_step(
            run_id=run_id,
            step_id=observation.step_id,
            payloads=trace_payloads,
        )
        return {
            **trace_payloads,
            "trace_path": str(step_dir),
        }

    @staticmethod
    def _can_execute(
        *,
        mode: AgentMode,
        approved: bool,
        safety_passed: bool,
    ) -> bool:
        if not safety_passed:
            return False
        if mode in {AgentMode.OBSERVE, AgentMode.PROPOSE}:
            return False
        if mode == AgentMode.APPROVAL:
            return approved
        return mode == AgentMode.CONSTRAINED_AUTO
