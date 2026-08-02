from __future__ import annotations

import json
import os
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


def _strip_ddp_prefix(state_dict: dict) -> dict:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _normalize_proprio(proprio: np.ndarray, stats: dict) -> np.ndarray:
    if "q01" in stats and "q99" in stats:
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
    elif "min" in stats and "max" in stats:
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
    else:
        raise KeyError("Proprio statistics must contain q01/q99 or min/max")
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (proprio - low) / (high - low + 1e-8) - 1.0, proprio)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


class OpenVLAOfflineRuntime:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        configure_offline_environment()
        self.layout: CheckpointLayout = inspect_checkpoint(checkpoint_dir)
        apply_bidirectional_attention_patch()
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for OpenVLA-OFT+ inference")
        self._torch = torch
        self._device = torch.device("cuda:0")

        for register in (
            lambda: AutoConfig.register("openvla", OpenVLAConfig),
            lambda: AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor),
            lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
            lambda: AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction),
        ):
            try:
                register()
            except ValueError:
                pass

        root = str(self.layout.root)
        self._processor = AutoProcessor.from_pretrained(
            root, trust_remote_code=True, local_files_only=True
        )
        self._vla = AutoModelForVision2Seq.from_pretrained(
            root,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(self._device)
        self._vla.vision_backbone.set_num_images_in_input(2)
        self._vla.eval()

        with self.layout.dataset_statistics.open("r", encoding="utf-8") as handle:
            self._vla.norm_stats = json.load(handle)
        preferred_key = os.environ.get("PARC_UNNORM_KEY", "libero_spatial")
        if preferred_key not in self._vla.norm_stats:
            available = sorted(self._vla.norm_stats)
            if not available:
                raise ValueError("dataset_statistics.json contains no normalization keys")
            preferred_key = available[0]
        self._unnorm_key = preferred_key

        self._proprio_projector = ProprioProjector(
            llm_dim=self._vla.llm_dim, proprio_dim=PROPRIO_DIM
        ).to(dtype=torch.bfloat16, device=self._device)
        proprio_state = torch.load(
            self.layout.proprio_projector, weights_only=True, map_location="cpu"
        )
        self._proprio_projector.load_state_dict(_strip_ddp_prefix(proprio_state))
        self._proprio_projector.eval()

        self._action_head = L1RegressionActionHead(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=ACTION_DIM,
        ).to(dtype=torch.bfloat16, device=self._device)
        action_state = torch.load(self.layout.action_head, weights_only=True, map_location="cpu")
        self._action_head.load_state_dict(_strip_ddp_prefix(action_state))
        self._action_head.eval()

    def predict_chunk(self, policy_input: PolicyInput) -> np.ndarray:
        torch = self._torch
        prompt = f"In: What action should the robot take to {policy_input.instruction.lower()}?\nOut:"
        inputs = self._processor(prompt, policy_input.full_image).to(
            self._device, dtype=torch.bfloat16
        )
        wrist_inputs = self._processor(prompt, policy_input.wrist_image).to(
            self._device, dtype=torch.bfloat16
        )
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )

        proprio_stats = self._vla.norm_stats[self._unnorm_key]["proprio"]
        proprio = _normalize_proprio(policy_input.proprio, proprio_stats)

        with torch.inference_mode():
            action, _ = self._vla.predict_action(
                **inputs,
                unnorm_key=self._unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self._proprio_projector,
                noisy_action_projector=None,
                action_head=self._action_head,
                use_film=False,
            )
        chunk = np.asarray(action, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"Expected model action chunk (T, 7), got {chunk.shape}")
        return chunk
