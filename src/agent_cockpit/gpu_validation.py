"""実CheckpointとRLDSを使うColab GPU検証ハーネス。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol

import numpy as np

from .dataset_explorer import EpisodeSample
from .policy_adapter import InferenceResult, compare_action_chunks


class EpisodeReader(Protocol):
    """GPU検証に必要なDataset Readerの最小契約。"""

    dataset_dir: Path

    def available_splits(self) -> list[str]: ...

    def read_sample(
        self,
        *,
        split: str,
        episode_offset: int,
        frame_id: int,
    ) -> EpisodeSample: ...


class PolicyAdapter(Protocol):
    """GPU検証に必要なPolicy Adapterの最小契約。"""

    checkpoint_dir: Path

    def predict_rlds(
        self,
        *,
        front_image: Any,
        wrist_image: Any,
        state: Any,
        instruction: str,
    ) -> InferenceResult: ...


@dataclass(frozen=True)
class GPUValidationConfig:
    """実データ推論の検証条件。"""

    split: str = "val"
    episode_offset: int = 0
    start_frame: int = 0
    num_frames: int = 3
    warmup_runs: int = 1
    require_cuda: bool = True

    def __post_init__(self) -> None:
        if not self.split.strip():
            raise ValueError("split must not be empty")
        if self.episode_offset < 0:
            raise ValueError("episode_offset must be non-negative")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_runtime_environment(*, require_cuda: bool) -> tuple[dict[str, Any], Any]:
    """PyTorch/CUDA環境を取得し、GPU必須条件を検証する。"""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU validation") from exc

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is False")

    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
            "total_memory_gib": round(properties.total_memory / 1024**3, 3),
        }

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "gpu": gpu,
        "packages": {
            "transformers": _package_version("transformers"),
            "tensorflow": _package_version("tensorflow"),
            "tensorflow-datasets": _package_version("tensorflow-datasets"),
            "gradio": _package_version("gradio"),
            "numpy": _package_version("numpy"),
        },
    }
    return environment, torch


def _memory_snapshot(torch: Any) -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


@dataclass
class ColabGPUValidationRunner:
    """実Checkpointを実RLDS Sampleへ適用し、再現可能な検証レポートを作る。"""

    reader: EpisodeReader
    policy: PolicyAdapter
    config: GPUValidationConfig

    def _predict(self, sample: EpisodeSample) -> InferenceResult:
        return self.policy.predict_rlds(
            front_image=sample.front_image,
            wrist_image=sample.wrist_image,
            state=sample.state,
            instruction=sample.instruction,
        )

    def run(self) -> dict[str, Any]:
        environment, torch = collect_runtime_environment(
            require_cuda=self.config.require_cuda
        )
        available_splits = self.reader.available_splits()
        if self.config.split not in available_splits:
            raise KeyError(
                f"Unknown split {self.config.split!r}; available={available_splits}"
            )

        first_sample = self.reader.read_sample(
            split=self.config.split,
            episode_offset=self.config.episode_offset,
            frame_id=self.config.start_frame,
        )

        for _ in range(self.config.warmup_runs):
            self._predict(first_sample)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
        memory_before = _memory_snapshot(torch)

        rows: list[dict[str, Any]] = []
        for index in range(self.config.num_frames):
            frame_id = self.config.start_frame + index
            try:
                sample = self.reader.read_sample(
                    split=self.config.split,
                    episode_offset=self.config.episode_offset,
                    frame_id=frame_id,
                )
            except IndexError:
                break

            result = self._predict(sample)
            metrics = compare_action_chunks(result.action_chunk, sample.action_chunk)
            finite = bool(np.isfinite(result.action_chunk).all())
            rows.append(
                {
                    "frame_id": frame_id,
                    "episode_id": sample.episode_id,
                    "instruction": sample.instruction,
                    "action_chunk_shape": list(result.action_chunk.shape),
                    "latency_ms": float(result.latency_ms),
                    "finite_action_chunk": finite,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mae_per_axis": metrics["mae_per_axis"],
                    "rmse_per_axis": metrics["rmse_per_axis"],
                }
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory_after = _memory_snapshot(torch)

        if not rows:
            raise RuntimeError("No RLDS frames were validated")
        latencies = [float(row["latency_ms"]) for row in rows]
        maes = [float(row["mae"]) for row in rows]
        finite_passed = all(bool(row["finite_action_chunk"]) for row in rows)
        shape_passed = all(row["action_chunk_shape"][-1] == 7 for row in rows)

        report = {
            "status": "pass" if finite_passed and shape_passed else "fail",
            "validation_type": "colab_gpu_real_checkpoint_rlds",
            "config": asdict(self.config),
            "checkpoint_dir": str(self.policy.checkpoint_dir),
            "dataset_dir": str(self.reader.dataset_dir),
            "environment": environment,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "checks": {
                "cuda_requirement_satisfied": (
                    environment["cuda_available"] or not self.config.require_cuda
                ),
                "samples_validated": len(rows),
                "finite_action_chunks": finite_passed,
                "action_dim_7": shape_passed,
            },
            "summary": {
                "validated_frames": len(rows),
                "mean_latency_ms": mean(latencies),
                "median_latency_ms": median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "mean_action_chunk_mae": mean(maes),
                "peak_allocated_gib": (
                    round(memory_after["peak_allocated_bytes"] / 1024**3, 3)
                    if memory_after["peak_allocated_bytes"] is not None
                    else None
                ),
                "peak_reserved_gib": (
                    round(memory_after["peak_reserved_bytes"] / 1024**3, 3)
                    if memory_after["peak_reserved_bytes"] is not None
                    else None
                ),
            },
            "samples": rows,
            "official_evaluation_data_used_for_training": False,
        }
        return report


def write_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    """検証結果を原子的にJSON保存する。"""

    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved
