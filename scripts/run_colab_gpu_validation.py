#!/usr/bin/env python3
"""Google Colabで実Checkpointと実RLDSのGPU推論を検証する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent_cockpit.dataset_explorer import RLDSEpisodeReader
from src.agent_cockpit.gpu_validation import (
    ColabGPUValidationRunner,
    GPUValidationConfig,
    write_validation_report,
)
from src.agent_cockpit.policy_adapter import OpenVLAPolicyAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the actual OpenVLA-OFT checkpoint against converted RLDS "
            "samples and persist latency, VRAM, shape, and action-error metrics."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="CUDA必須Gateを解除する。実Checkpoint検証では通常使用しない。",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reader = RLDSEpisodeReader(args.dataset_dir)
    policy = OpenVLAPolicyAdapter(args.checkpoint_dir)
    runner = ColabGPUValidationRunner(
        reader=reader,
        policy=policy,
        config=GPUValidationConfig(
            split=args.split,
            episode_offset=args.episode_offset,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            warmup_runs=args.warmup_runs,
            require_cuda=not args.allow_cpu,
        ),
    )
    report = runner.run()
    output = write_validation_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "summary": report["summary"],
                "checks": report["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
