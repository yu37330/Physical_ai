"""OpenVLA-OFT推論をAgent Cockpitへ接続するアダプター。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .models import ActionCandidate, ActionProposal, AgentObservation


class ActionChunkRuntime(Protocol):
    """OpenVLA runtimeが満たす最小契約。"""

    def predict_chunk(self, policy_input: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class InferenceResult:
    """単発Action chunk推論結果。"""

    action_chunk: np.ndarray
    latency_ms: float
    checkpoint_dir: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_chunk": self.action_chunk.astype(float).tolist(),
            "shape": list(self.action_chunk.shape),
            "latency_ms": self.latency_ms,
            "checkpoint_dir": self.checkpoint_dir,
            "source": self.source,
        }


def _validate_rgb_image(image: Any, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_center_crop_without_rotation(
    image: Any,
    *,
    size: int = 224,
    crop_area: float = 0.9,
) -> Image.Image:
    """RLDS保存済み画像を再回転せず、OpenVLA入力サイズへ合わせる。"""

    array = _validate_rgb_image(image, name="image")
    pil = Image.fromarray(array, mode="RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    scale = math.sqrt(crop_area)
    crop_width = max(1, int(round(size * scale)))
    crop_height = max(1, int(round(size * scale)))
    left = (size - crop_width) // 2
    top = (size - crop_height) // 2
    return pil.crop((left, top, left + crop_width, top + crop_height)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


class OpenVLAPolicyAdapter:
    """既存の提出用OpenVLAOfflineRuntimeをUIから再利用する。"""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        runtime: ActionChunkRuntime | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        if runtime is None:
            if not self.checkpoint_dir.is_dir():
                raise FileNotFoundError(
                    f"OpenVLA checkpoint directory not found: {self.checkpoint_dir}"
                )
            from submission.openvla_oft_offline.runtime.model_runtime import (
                OpenVLAOfflineRuntime,
            )

            runtime = OpenVLAOfflineRuntime(self.checkpoint_dir)
        self._runtime = runtime

    def predict_rlds(
        self,
        *,
        front_image: Any,
        wrist_image: Any,
        state: Any,
        instruction: str,
    ) -> InferenceResult:
        """変換済みRLDSの画像方向と8次元Stateを保ったまま推論する。"""

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (8,):
            raise ValueError(f"Expected RLDS state (8,), got {state_array.shape}")
        if not np.isfinite(state_array).all():
            raise ValueError("state contains NaN or Inf")
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        from submission.openvla_oft_offline.runtime.preprocessing import PolicyInput

        policy_input = PolicyInput(
            full_image=_resize_center_crop_without_rotation(front_image),
            wrist_image=_resize_center_crop_without_rotation(wrist_image),
            proprio=state_array,
            instruction=instruction.strip(),
        )
        started = perf_counter()
        chunk = np.asarray(
            self._runtime.predict_chunk(policy_input), dtype=np.float32
        )
        latency_ms = (perf_counter() - started) * 1000.0

        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"Expected action chunk (T, 7), got {chunk.shape}")
        if chunk.shape[0] == 0:
            raise ValueError("OpenVLA returned an empty action chunk")
        if not np.isfinite(chunk).all():
            raise ValueError("OpenVLA action chunk contains NaN or Inf")

        return InferenceResult(
            action_chunk=chunk,
            latency_ms=latency_ms,
            checkpoint_dir=str(self.checkpoint_dir),
            source="openvla_oft_offline_runtime",
        )


def compare_action_chunks(predicted: Any, target: Any) -> dict[str, Any]:
    """予測と教師Action chunkを共通長で比較する。"""

    predicted_array = np.asarray(predicted, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    for name, array in (("predicted", predicted_array), ("target", target_array)):
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"{name} must have shape (T, 7), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    common_steps = min(predicted_array.shape[0], target_array.shape[0])
    if common_steps == 0:
        raise ValueError("Action chunks must contain at least one step")
    error = predicted_array[:common_steps] - target_array[:common_steps]
    mae_per_axis = np.mean(np.abs(error), axis=0)
    rmse_per_axis = np.sqrt(np.mean(np.square(error), axis=0))
    return {
        "common_steps": common_steps,
        "predicted_steps": int(predicted_array.shape[0]),
        "target_steps": int(target_array.shape[0]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae_per_axis": mae_per_axis.astype(float).tolist(),
        "rmse_per_axis": rmse_per_axis.astype(float).tolist(),
    }


class OpenVLAPolicyPlanner:
    """OpenVLAのAction chunk先頭を次Actionとして提案するPlanner。"""

    def __init__(self, policy: OpenVLAPolicyAdapter) -> None:
        self.policy = policy

    def propose(self, observation: AgentObservation, goal: str) -> ActionProposal:
        if not observation.image_path:
            raise ValueError("front image path is required for OpenVLA planning")
        wrist_image_path = str(observation.metadata.get("wrist_image_path", ""))
        if not wrist_image_path:
            raise ValueError(
                "observation.metadata['wrist_image_path'] is required for OpenVLA planning"
            )

        front = np.asarray(Image.open(observation.image_path).convert("RGB"))
        wrist = np.asarray(Image.open(wrist_image_path).convert("RGB"))
        result = self.policy.predict_rlds(
            front_image=front,
            wrist_image=wrist,
            state=observation.state,
            instruction=observation.instruction,
        )
        first_action = result.action_chunk[0].astype(float).tolist()
        selected_id = "openvla_chunk_step_0"
        candidates = [
            ActionCandidate(
                action_id=selected_id,
                label="OpenVLA Action chunk先頭",
                score=1.0,
                action=first_action,
                expected_result="OpenVLA Policyが予測した次状態へ遷移する",
                source="openvla_oft",
            ),
            ActionCandidate(
                action_id="hold_and_reobserve",
                label="停止して再観測",
                score=0.0,
                action=[0.0] * 7,
                expected_result="環境を動かさず観測を更新する",
                source="safety_fallback",
            ),
        ]
        return ActionProposal(
            goal=goal.strip(),
            subgoal=observation.instruction.strip(),
            candidates=candidates,
            selected_action_id=selected_id,
            uncertainty=-1.0,
            requires_approval=True,
            metadata={
                "confidence_available": False,
                "action_chunk": result.action_chunk.astype(float).tolist(),
                "action_chunk_shape": list(result.action_chunk.shape),
                "latency_ms": result.latency_ms,
                "checkpoint_dir": result.checkpoint_dir,
            },
        )
