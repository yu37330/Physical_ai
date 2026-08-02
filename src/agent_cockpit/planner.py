"""High-level Plannerの最小実装。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ActionCandidate, ActionProposal, AgentObservation


@dataclass(frozen=True)
class RuleBasedPlanner:
    """GPUモデル未接続でもUIとTraceを検証できるPlanner。

    実運用では、このクラスと同じ ``propose`` インターフェースを持つ
    VLM/LLM Plannerへ差し替える。
    """

    action_dim: int = 7

    def propose(self, observation: AgentObservation, goal: str) -> ActionProposal:
        if not goal.strip():
            raise ValueError("goal must not be empty")
        if len(observation.state) < self.action_dim:
            raise ValueError(
                f"state must contain at least {self.action_dim} values, got {len(observation.state)}"
            )

        instruction = observation.instruction.lower()
        if "open" in instruction or "開" in observation.instruction:
            subgoal = "把持位置を合わせて対象を開く"
            primary = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.2]
            expected = "対象の開度が増加する"
        elif "pick" in instruction or "grasp" in instruction or "取" in observation.instruction:
            subgoal = "対象物へ接近して把持姿勢を作る"
            primary = [0.02, 0.0, -0.01, 0.0, 0.0, 0.0, -0.2]
            expected = "グリッパーと対象物の距離が短縮する"
        else:
            subgoal = "観測を維持しながら安全な微小移動を試す"
            primary = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            expected = "対象に対する位置関係が小さく変化する"

        candidates = [
            ActionCandidate(
                action_id="policy_primary",
                label="Policy推奨Action",
                score=0.82,
                action=primary,
                expected_result=expected,
                source="rule_based_planner",
            ),
            ActionCandidate(
                action_id="observe_again",
                label="再観測",
                score=0.55,
                action=[0.0] * self.action_dim,
                expected_result="観測を更新して不確実性を下げる",
                source="rule_based_planner",
            ),
            ActionCandidate(
                action_id="safe_retreat",
                label="安全側へ退避",
                score=0.31,
                action=[-0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0],
                expected_result="対象物と周辺物から距離を確保する",
                source="rule_based_planner",
            ),
        ]
        return ActionProposal(
            goal=goal.strip(),
            subgoal=subgoal,
            candidates=candidates,
            selected_action_id="policy_primary",
            uncertainty=0.18,
            requires_approval=True,
        )
