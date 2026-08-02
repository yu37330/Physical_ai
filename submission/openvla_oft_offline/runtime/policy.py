from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from .action_postprocess import ActionChunkBuffer, finalize_action
from .model_runtime import ChunkPredictor, OpenVLAOfflineRuntime
from .preprocessing import build_policy_input


class OfflinePolicy:
    def __init__(self, checkpoint_dir: str | Path, runtime: ChunkPredictor | None = None) -> None:
        self._runtime = runtime or OpenVLAOfflineRuntime(checkpoint_dir)
        self._instruction = ""
        self._actions = ActionChunkBuffer()
        self._lock = Lock()

    def reset(self, instruction: str, seed: int | None = None) -> None:
        del seed
        with self._lock:
            self._instruction = str(instruction)
            self._actions.clear()

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        with self._lock:
            if self._actions.empty:
                policy_input = build_policy_input(observation, self._instruction)
                self._actions.load(self._runtime.predict_chunk(policy_input))
            return finalize_action(self._actions.pop())
