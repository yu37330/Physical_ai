from __future__ import annotations

from collections import deque

import numpy as np


class ActionChunkBuffer:
    def __init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque()

    def clear(self) -> None:
        self._buffer.clear()

    def load(self, action_chunk: np.ndarray) -> None:
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected action chunk shape (T, 7), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Action chunk contains NaN or Inf")
        self._buffer.clear()
        self._buffer.extend(action.copy() for action in actions)

    def pop(self) -> np.ndarray:
        if not self._buffer:
            raise IndexError("Action chunk buffer is empty")
        return np.asarray(self._buffer.popleft(), dtype=np.float32)

    @property
    def empty(self) -> bool:
        return not self._buffer


def finalize_action(action: np.ndarray, clip: float = 1.0) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).reshape(7)
    if not np.isfinite(result).all():
        raise ValueError("Action contains NaN or Inf")
    return np.clip(result, -clip, clip).astype(np.float32, copy=False)
