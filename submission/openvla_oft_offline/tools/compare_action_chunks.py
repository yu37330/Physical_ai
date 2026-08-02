from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    reference = np.load(args.reference)
    candidate = np.load(args.candidate)
    if reference.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: {reference.shape} != {candidate.shape}")

    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    report = {
        "shape": list(reference.shape),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "mean_abs_diff": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "atol": args.atol,
        "passed": bool(np.allclose(reference, candidate, rtol=0.0, atol=args.atol)),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
