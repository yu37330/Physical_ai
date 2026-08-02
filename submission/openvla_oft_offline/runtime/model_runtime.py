from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .bidirectional_attention import apply_bidirectional_attention_patch
from .checkpoint_layout import CheckpointLayout, inspect_checkpoint
from .offline_env import configure_offline_environment
from .preprocessing import PolicyInput


class ChunkPredictor(Protocol):
    def predict_chunk(self, policy_input: PolicyInput) -> np.ndarray:
        ...


class OpenVLAOfflineRuntime:
    """Submission runtime boundary.

    The concrete Prismatic/OpenVLA model connection is kept behind this class so
    preprocessing, API handling and checkpoint validation can be tested before
    the 15GB checkpoint is available.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        configure_offline_environment()
        self.layout: CheckpointLayout = inspect_checkpoint(checkpoint_dir)
        apply_bidirectional_attention_patch()
        self._model = self._load_model()

    def _load_model(self):
        raise NotImplementedError(
            "Vendor the minimal OpenVLA/Prismatic inference modules and load the "
            "local checkpoint with local_files_only=True."
        )

    def predict_chunk(self, policy_input: PolicyInput) -> np.ndarray:
        raise NotImplementedError("Connect the validated OpenVLA-OFT inference path")
