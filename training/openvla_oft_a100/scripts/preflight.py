from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _disk(path: Path) -> dict[str, int | str]:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {
        "requested_path": str(path),
        "checked_path": str(existing),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Colab/A100 environment before PARC work")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/PARC2026"))
    parser.add_argument("--minimum-work-free-gb", type=float, default=80.0)
    parser.add_argument("--minimum-drive-free-gb", type=float, default=30.0)
    parser.add_argument("--require-a100-40gb", action="store_true")
    parser.add_argument(
        "--no-require-drive",
        action="store_true",
        help="Report the Drive checks without requiring them. For runs that "
        "measure something and persist nothing.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    drive_mounted = Path("/content/drive/MyDrive").is_dir() and args.drive_root.is_dir()
    work_disk = _disk(args.work_root)
    drive_disk = _disk(args.drive_root)
    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    git = _run(["git", "-C", str(project_root), "rev-parse", "HEAD"])
    branch = _run(["git", "-C", str(project_root), "branch", "--show-current"])

    gpu_text = nvidia.get("stdout", "") if nvidia.get("returncode") == 0 else ""
    has_a100 = "A100" in gpu_text.upper()
    memory_values: list[int] = []
    for line in gpu_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                memory_values.append(int(parts[1]))
            except ValueError:
                pass
    has_40gb = any(value >= 39000 for value in memory_values)

    required_paths = [
        project_root / "training/openvla_oft_a100/scripts/bootstrap_colab.sh",
        project_root / "training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh",
        project_root / "submission/openvla_oft_offline/policy_server.py",
        project_root / "configs/models/openvla_oft_plus_checkpoint.yaml",
        project_root / "configs/datasets/mini_e2e_v001.yaml",
    ]
    path_checks = {str(path.relative_to(project_root)): path.is_file() for path in required_paths}

    checks = {
        "python_3_10_or_newer": sys.version_info[:2] >= (3, 10),
        "project_files_present": all(path_checks.values()),
        "git_commit_resolved": git.get("returncode") == 0 and bool(git.get("stdout")),
        "drive_mounted": drive_mounted,
        "work_disk_free": int(work_disk["free_bytes"]) >= args.minimum_work_free_gb * 1024**3,
        "drive_disk_free": drive_mounted
        and int(drive_disk["free_bytes"]) >= args.minimum_drive_free_gb * 1024**3,
        "nvidia_smi_available": nvidia.get("returncode") == 0,
        "a100_40gb": has_a100 and has_40gb,
    }
    required_check_names = [
        "python_3_10_or_newer",
        "project_files_present",
        "git_commit_resolved",
        "work_disk_free",
        "nvidia_smi_available",
    ]
    if not args.no_require_drive:
        required_check_names[3:3] = ["drive_mounted", "drive_disk_free"]
    if args.require_a100_40gb:
        required_check_names.append("a100_40gb")
    status = "pass" if all(checks[name] for name in required_check_names) else "fail"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "platform": platform.platform(),
        "python": sys.version,
        "environment": {
            "colab_release_tag": os.getenv("COLAB_RELEASE_TAG"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "git": {"commit": git, "branch": branch},
        "gpu": nvidia,
        "disk": {"work": work_disk, "drive": drive_disk},
        "required_paths": path_checks,
        "checks": checks,
        "required_checks": required_check_names,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if status != "pass":
        # The JSON alone makes a failure hard to read in a terminal, so restate
        # the failing gates with the numbers that caused them.
        failed = [name for name in required_check_names if not checks[name]]
        print("\nPreflight failed:", ", ".join(failed), file=sys.stderr)
        for name, disk, minimum in (
            ("work_disk_free", work_disk, args.minimum_work_free_gb),
            ("drive_disk_free", drive_disk, args.minimum_drive_free_gb),
        ):
            if name in failed:
                free_gb = int(disk["free_bytes"]) / 1024**3
                print(
                    f"  {name}: {free_gb:.1f}GB free at {disk['checked_path']},"
                    f" need {minimum:g}GB",
                    file=sys.stderr,
                )
        if "a100_40gb" in failed:
            print(f"  a100_40gb: {gpu_text or 'no GPU reported'}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
