#!/usr/bin/env python3
"""Base重みとStage Aの学習済みcomponentを合成し、提出用Checkpointを作る。

提出物は「公開重みを実質的に変更せず推論する」構成であってはならない
（docs/OFFICIAL_RULES.md 6章）。Stage Aで学習したAction HeadとProprio Projectorを
Base重みへ差し替えることで、独自学習要素がAction生成へ実質的に寄与する構成になる。

    python scripts/assemble_submission_checkpoint.py \\
      --base-checkpoint /content/work/models/openvla_oft_plus_base \\
      --trained-run-dir /content/work/runs/stage_a_s1_head_proprio_100 \\
      --output submission/openvla_oft_offline/model_weights/openvla_oft_plus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 学習状態やLoRA adapterは提出物へ含めない。
# configs/models/openvla_oft_plus_checkpoint.yaml の submission_exclude_patterns と対応する。
EXCLUDE_NAMES = ("optimizer", "scheduler", "trainer_state", "rng_state")
EXCLUDE_DIRS = (".cache", "lora_adapter", "checkpoint-", "runs", "wandb")

STEP_PATTERN = re.compile(r"--(\d+)_checkpoint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest(paths: list[Path]) -> Path:
    """Highest training step, so a run that kept several checkpoints is unambiguous."""

    def step(path: Path) -> int:
        match = STEP_PATTERN.search(path.name)
        return int(match.group(1)) if match else -1

    return max(paths, key=step)


def find_trained_component(run_dir: Path, pattern: str) -> Path:
    matches = [path for path in run_dir.rglob(pattern) if path.is_file()]
    if not matches:
        raise SystemExit(f"No {pattern} under {run_dir}. Did Stage A save a checkpoint?")
    return _latest(matches)


def _skip(relative: Path) -> bool:
    if any(part.startswith(EXCLUDE_DIRS) for part in relative.parts):
        return True
    return relative.name.startswith(EXCLUDE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    base = args.base_checkpoint.resolve()
    run_dir = args.trained_run_dir.resolve()
    output = args.output.resolve()
    if not base.is_dir():
        raise SystemExit(f"Base checkpoint not found: {base}")
    if not run_dir.is_dir():
        raise SystemExit(f"Stage A run directory not found: {run_dir}")

    action_head = find_trained_component(run_dir, "*action_head*checkpoint*.pt")
    proprio = find_trained_component(run_dir, "*proprio_projector*checkpoint*.pt")
    # The action head learned to emit actions normalised against the statistics of
    # the dataset it trained on, so the submission has to unnormalise with those
    # rather than the base checkpoint's.
    statistics = [path for path in run_dir.rglob("dataset_statistics.json") if path.is_file()]
    if not statistics:
        raise SystemExit(f"No dataset_statistics.json under {run_dir}")
    trained_statistics = statistics[0]

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if _skip(relative):
            continue
        # Replaced below; leaving the base ones would break inspect_checkpoint,
        # which requires exactly one of each.
        if "action_head" in relative.name or "proprio_projector" in relative.name:
            continue
        if relative.name == "dataset_statistics.json":
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1

    shutil.copy2(action_head, output / action_head.name)
    shutil.copy2(proprio, output / proprio.name)
    shutil.copy2(trained_statistics, output / "dataset_statistics.json")

    from submission.openvla_oft_offline.runtime.checkpoint_layout import inspect_checkpoint

    layout = inspect_checkpoint(output)

    source_manifest = base / "model_source_manifest.json"
    base_revision = None
    if source_manifest.is_file():
        base_revision = json.loads(source_manifest.read_text(encoding="utf-8")).get(
            "resolved_revision"
        )

    report: dict[str, Any] = {
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "base_checkpoint": str(base),
        "base_resolved_revision": base_revision,
        "trained_run_dir": str(run_dir),
        "trained_components": {
            "action_head": {"source": str(action_head), "sha256": sha256(action_head)},
            "proprio_projector": {"source": str(proprio), "sha256": sha256(proprio)},
        },
        "dataset_statistics_source": str(trained_statistics),
        "dataset_statistics_keys": sorted(
            json.loads(trained_statistics.read_text(encoding="utf-8"))
        ),
        "base_files_copied": copied,
        "total_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "layout": {
            "model_files": [path.name for path in layout.model_files],
            "action_head": layout.action_head.name,
            "proprio_projector": layout.proprio_projector.name,
        },
    }
    (output / "parc_submission_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
