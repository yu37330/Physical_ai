"""ActionをExecutorへ渡す前の安全制約。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ActionCandidate, SafetyResult


@dataclass(frozen=True)
class ActionSafetyValidator:
    """Action次元、有限値、軸ごとの変位、グリッパー範囲を確認する。"""

    action_dim: int = 7
    max_abs_delta: float = 0.25
    max_gripper_abs: float = 1.0

    def validate(self, candidate: ActionCandidate) -> SafetyResult:
        violations: list[str] = []
        action = candidate.action

        dimension_ok = len(action) == self.action_dim
        if not dimension_ok:
            violations.append(
                f"action dimension must be {self.action_dim}, got {len(action)}"
            )

        finite_ok = all(math.isfinite(value) for value in action)
        if not finite_ok:
            violations.append("action contains NaN or Inf")

        delta_ok = dimension_ok and all(
            abs(value) <= self.max_abs_delta for value in action[:-1]
        )
        if dimension_ok and not delta_ok:
            violations.append("motion delta exceeds configured limit")

        gripper_ok = dimension_ok and abs(action[-1]) <= self.max_gripper_abs
        if dimension_ok and not gripper_ok:
            violations.append("gripper command exceeds configured limit")

        checks = {
            "dimension": dimension_ok,
            "finite": finite_ok,
            "motion_delta": delta_ok,
            "gripper": gripper_ok,
        }
        return SafetyResult(
            passed=all(checks.values()),
            checks=checks,
            violations=violations,
        )
