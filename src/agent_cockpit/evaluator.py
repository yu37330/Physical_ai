"""期待状態と実状態の比較を行う最小Evaluator。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import EvaluationResult


@dataclass(frozen=True)
class StateDeltaEvaluator:
    """状態ベクトルの変化量から進捗を暫定評価する。"""

    success_delta: float = 0.05
    partial_delta: float = 0.005

    def evaluate(
        self,
        before_state: list[float],
        after_state: list[float] | None,
    ) -> EvaluationResult:
        if after_state is None:
            return EvaluationResult(
                status="pending",
                progress=0.0,
                summary="実行後Stateが未入力のため評価待ちです。",
                should_replan=False,
            )
        if len(before_state) != len(after_state):
            raise ValueError("before_state and after_state must have the same length")

        delta = math.sqrt(
            sum((after - before) ** 2 for before, after in zip(before_state, after_state))
        )
        if delta >= self.success_delta:
            return EvaluationResult(
                status="progressed",
                progress=min(delta / self.success_delta, 1.0),
                summary=f"State変化量 {delta:.4f}。期待方向の確認後、次サブゴールへ進めます。",
                should_replan=True,
            )
        if delta >= self.partial_delta:
            return EvaluationResult(
                status="partial",
                progress=min(delta / self.success_delta, 1.0),
                summary=f"State変化量 {delta:.4f}。追加Actionまたは再観測が必要です。",
                should_replan=True,
            )
        return EvaluationResult(
            status="stalled",
            progress=0.0,
            summary=f"State変化量 {delta:.4f}。同一Action反復を避けて再計画します。",
            should_replan=True,
        )
