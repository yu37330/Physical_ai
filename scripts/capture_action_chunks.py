#!/usr/bin/env python3
"""固定観測に対するAction chunkを`.npy`へ保存する。

`compare_action_chunks.py`は2つの配列を比較する実装があるが、比較対象を生成する
側が無かった。このScriptがそれを埋める。

`OPENVLA_OFFLINE_RUNTIME_ASSESSMENT.md` 8章「PyPI Transformers＋Patchと公式Forkの
Action一致」を実行するには、次の2つを取って突き合わせる。

    # 参考実装: 公式Fork。Forkは双方向Attentionを内蔵しているので自前Patchは当てない
    python scripts/capture_action_chunks.py --checkpoint-dir ... \\
      --no-attention-patch --output fork.npy

    # 提出構成: PyPI transformers + 自前Patch
    python scripts/capture_action_chunks.py --checkpoint-dir ... --output pypi.npy

    python submission/openvla_oft_offline/tools/compare_action_chunks.py fork.npy pypi.npy

観測は固定Seedの合成データで、実行間で完全に再現する。実LIBERO画像ではないため、
実装の食い違いは検出できるが、前処理の微小な差までは代表しない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Notebooks and wrappers invoke this as `python scripts/capture_action_chunks.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from submission.openvla_oft_offline.runtime.model_runtime import (  # noqa: E402
    OpenVLAOfflineRuntime,
)
from submission.openvla_oft_offline.runtime.preprocessing import (  # noqa: E402
    build_policy_input,
)

CAMERA = 128


def build_observations(count: int, seed: int, camera: int) -> list[dict[str, np.ndarray]]:
    """運営validatorと同じキー・shape・dtype。Seed固定で実行間に再現する。"""
    rng = np.random.default_rng(seed)
    return [
        {
            "agentview_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
            "robot0_joint_pos": rng.uniform(-2.0, 2.0, 7).astype(np.float32),
            "robot0_eef_pos": rng.uniform(-0.5, 0.5, 3).astype(np.float32),
            "robot0_eef_quat": rng.uniform(-1.0, 1.0, 4).astype(np.float32),
            "robot0_gripper_qpos": rng.uniform(-0.05, 0.05, 2).astype(np.float32),
        }
        for _ in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--camera", type=int, default=CAMERA)
    parser.add_argument(
        "--instruction",
        default="pick up the black bowl and place it on the plate",
    )
    parser.add_argument(
        "--no-attention-patch",
        action="store_true",
        help="Skip the bidirectional attention patch. Use when capturing the "
        "OpenVLA-OFT fork's native behaviour as the parity reference.",
    )
    args = parser.parse_args()

    runtime = OpenVLAOfflineRuntime(
        args.checkpoint_dir, apply_attention_patch=not args.no_attention_patch
    )

    chunks = [
        runtime.predict_chunk(build_policy_input(observation, args.instruction))
        for observation in build_observations(args.num_frames, args.seed, args.camera)
    ]
    shapes = {chunk.shape for chunk in chunks}
    if len(shapes) != 1:
        raise SystemExit(f"Action chunks have inconsistent shapes: {sorted(shapes)}")

    stacked = np.stack(chunks).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, stacked)

    import transformers
    from importlib.metadata import distribution

    # Both builds report 4.40.1, so the version alone cannot show that the
    # reference and candidate runs used different transformers. Without this the
    # comparison could pass simply because nothing was swapped.
    from_git_fork = distribution("transformers").read_text("direct_url.json") is not None

    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(stacked.shape),
                "attention_patch_applied": not args.no_attention_patch,
                "transformers": transformers.__version__,
                "transformers_is_fork": from_git_fork,
                "seed": args.seed,
                "finite": bool(np.isfinite(stacked).all()),
                "abs_max": float(np.abs(stacked).max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
