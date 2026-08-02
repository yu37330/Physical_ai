"""Agent Cockpitで共有するドメインモデル。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AgentMode(str, Enum):
    """自律実行レベル。"""

    OBSERVE = "observe"
    PROPOSE = "propose"
    APPROVAL = "approval"
    CONSTRAINED_AUTO = "constrained_auto"


@dataclass(frozen=True)
class AgentObservation:
    """1ステップ分の観測。画像本体は保存せず参照パスを持つ。"""

    step_id: int
    instruction: str
    state: list[float]
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionCandidate:
    """PlannerまたはPolicyが生成した候補Action。"""

    action_id: str
    label: str
    score: float
    action: list[float]
    expected_result: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionProposal:
    """次のサブゴールと候補Actionをまとめた提案。"""

    goal: str
    subgoal: str
    candidates: list[ActionCandidate]
    selected_action_id: str
    uncertainty: float
    requires_approval: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> ActionCandidate:
        for candidate in self.candidates:
            if candidate.action_id == self.selected_action_id:
                return candidate
        raise ValueError(f"Selected action not found: {self.selected_action_id}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected"] = self.selected.to_dict()
        return payload


@dataclass(frozen=True)
class SafetyResult:
    """Action制約の検証結果。"""

    passed: bool
    checks: dict[str, bool]
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Action実行後の評価結果。"""

    status: str
    progress: float
    summary: str
    should_replan: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
