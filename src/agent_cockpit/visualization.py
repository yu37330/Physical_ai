"""Agent CockpitのAction chunk可視化。"""

from __future__ import annotations

from typing import Any

import numpy as np


def action_chunk_rows(
    chunk: Any,
    *,
    prefix: str = "predicted",
) -> list[list[float | int | str]]:
    """Action chunkをGradio Dataframe向けの行へ変換する。"""

    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"Expected action chunk (T, 7), got {array.shape}")
    return [
        [prefix, int(step), *array[step].astype(float).tolist()]
        for step in range(array.shape[0])
    ]


def action_chunk_figure(predicted: Any, target: Any | None = None) -> Any:
    """予測Action chunkと教師Action chunkを軸別に描画する。"""

    import matplotlib.pyplot as plt

    predicted_array = np.asarray(predicted, dtype=np.float32)
    if predicted_array.ndim != 2 or predicted_array.shape[1] != 7:
        raise ValueError(
            f"Expected predicted action chunk (T, 7), got {predicted_array.shape}"
        )

    target_array: np.ndarray | None = None
    if target is not None:
        target_array = np.asarray(target, dtype=np.float32)
        if target_array.ndim != 2 or target_array.shape[1] != 7:
            raise ValueError(
                f"Expected target action chunk (T, 7), got {target_array.shape}"
            )

    figure, axis = plt.subplots(figsize=(10, 5))
    predicted_steps = np.arange(predicted_array.shape[0])
    for action_axis in range(7):
        axis.plot(
            predicted_steps,
            predicted_array[:, action_axis],
            marker="o",
            label=f"pred a{action_axis}",
        )
        if target_array is not None:
            target_steps = np.arange(target_array.shape[0])
            axis.plot(
                target_steps,
                target_array[:, action_axis],
                linestyle="--",
                alpha=0.65,
                label=f"target a{action_axis}",
            )
    axis.set_xlabel("Action chunk step")
    axis.set_ylabel("Action value")
    axis.set_title("Predicted / target action chunk")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    return figure
