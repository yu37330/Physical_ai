#!/usr/bin/env python3
"""OpenVLA-OFT CheckpointをLIBERO環境で1 Trial閉ループ評価する。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Notebooks invoke this as `python scripts/run_libero_closed_loop.py`, which puts
# `scripts/` on sys.path instead of the repo root. Add the root so `src` and
# `submission` resolve without requiring PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_cockpit.libero_closed_loop import (
    LiberoClosedLoopConfig,
    LiberoClosedLoopRunner,
)
from src.agent_cockpit.libero_executor import LiberoSimulationExecutor, LiberoTaskConfig
from src.agent_cockpit.storage import TraceStore
from submission.openvla_oft_offline.runtime.policy import OfflinePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one causal OpenVLA-OFT trial in the official LIBERO "
            "OffScreenRenderEnv and persist a structured evaluation trace."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--task-suite-name",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        required=True,
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-abs-delta", type=float, default=1.0)
    parser.add_argument("--persist-observation-images", action="store_true")
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=None,
        help=(
            "Run trace root. Omit to use "
            "$PHYSICAL_AI_DRIVE_ROOT/40_experiments/agent_cockpit."
        ),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    run_id = args.run_id.strip() or datetime.now().strftime(
        "libero_%Y%m%d_%H%M%S"
    )
    trace_store = (
        TraceStore(args.trace_root.expanduser())
        if args.trace_root is not None
        else TraceStore.from_environment()
    )
    executor = LiberoSimulationExecutor(
        LiberoTaskConfig(
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            seed=args.seed,
            resolution=args.resolution,
            settle_steps=args.settle_steps,
            max_steps=args.max_steps,
        )
    )
    runner = LiberoClosedLoopRunner(
        executor=executor,
        policy=OfflinePolicy(args.checkpoint_dir),
        trace_store=trace_store,
        config=LiberoClosedLoopConfig(
            max_steps=args.max_steps,
            persist_observation_images=args.persist_observation_images,
            max_abs_delta=args.max_abs_delta,
        ),
    )
    summary = runner.run(run_id=run_id)
    output = args.output or Path(summary["summary_path"])
    if output != Path(summary["summary_path"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "success": summary["success"],
                "stop_reason": summary["stop_reason"],
                "executed_steps": summary["executed_steps"],
                "summary_path": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
