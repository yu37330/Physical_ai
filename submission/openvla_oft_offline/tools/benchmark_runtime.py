from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.model_runtime import OpenVLAOfflineRuntime
from runtime.preprocessing import build_policy_input


def load_observation(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def timed_predict(runtime: OpenVLAOfflineRuntime, policy_input) -> tuple[np.ndarray, float]:
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    actions = runtime.predict_chunk(policy_input)
    torch.cuda.synchronize()
    return actions, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--instruction", default="move the black bowl to the plate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions-output", type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime = OpenVLAOfflineRuntime(args.model_dir)
    torch.cuda.synchronize()
    cold_start_seconds = time.perf_counter() - load_started

    observation = load_observation(args.fixture)
    policy_input = build_policy_input(observation, args.instruction)
    first_actions, first_seconds = timed_predict(runtime, policy_input)

    for _ in range(args.warmup):
        timed_predict(runtime, policy_input)

    timings: list[float] = []
    last_actions = first_actions
    for _ in range(args.repeats):
        last_actions, elapsed = timed_predict(runtime, policy_input)
        timings.append(elapsed)

    if last_actions.ndim != 2 or last_actions.shape[1] != 7:
        raise SystemExit(f"Unexpected action shape: {last_actions.shape}")
    if not np.isfinite(last_actions).all():
        raise SystemExit("Action chunk contains NaN or Inf")

    sorted_timings = sorted(timings)
    p95_index = max(0, min(len(sorted_timings) - 1, int(np.ceil(0.95 * len(sorted_timings))) - 1))
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cold_start_seconds": cold_start_seconds,
        "first_inference_seconds": first_seconds,
        "warm_inference": {
            "count": len(timings),
            "mean_seconds": statistics.fmean(timings),
            "median_seconds": statistics.median(timings),
            "p95_seconds": sorted_timings[p95_index],
            "max_seconds": max(timings),
        },
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "action_shape": list(last_actions.shape),
        "action_min": float(last_actions.min()),
        "action_max": float(last_actions.max()),
        "passed_internal_latency_gate": max([first_seconds, *timings]) < 8.0,
        "passed_internal_vram_gate": torch.cuda.max_memory_allocated() < 22 * 1024**3,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.actions_output is not None:
        args.actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.actions_output, last_actions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
