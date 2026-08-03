"""Physical AI agent cockpit domain package。"""

from __future__ import annotations

from typing import Any

from .models import (
    ActionCandidate,
    ActionProposal,
    AgentMode,
    AgentObservation,
    EvaluationResult,
    SafetyResult,
)

__all__ = [
    "ActionCandidate",
    "ActionProposal",
    "AgentMode",
    "AgentObservation",
    "EvaluationResult",
    "SafetyResult",
    "AgentOrchestrator",
]


def __getattr__(name: str) -> Any:
    """重いUI・保存依存を、実際に必要になるまで読み込まない。"""

    if name == "AgentOrchestrator":
        from .orchestrator import AgentOrchestrator

        return AgentOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
