from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from build_checkpoint_manifest import build_manifest


def _materialize_symlinks(root: Path) -> list[str]:
    materialized: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        relative = path.relative_to(root).as_posix()
        temporary = path.with_name(path.name + ".materializing")
        shutil.copy2(target, temporary)
        path.unlink()
        temporary.replace(path)
        materialized.append(relative)
    return materialized


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Download and inventory the pinned OpenVLA-OFT+ checkpoint")
    parser.add_argument(
        "--repo-id",
        default="Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest-config",
        type=Path,
        default=project_root / "configs/models/openvla_oft_plus_checkpoint.yaml",
    )
    parser.add_argument("--allow-manifest-fail", action="store_true")
    args = parser.parse_args()

    info = HfApi().model_info(args.repo_id, revision=args.revision)
    resolved_sha = info.sha
    output = args.output.expanduser().resolve()
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved_sha,
        local_dir=output,
    )
    materialized_symlinks = _materialize_symlinks(output)

    source_manifest = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_sha,
        "materialized_symlinks": materialized_symlinks,
    }
    source_manifest_path = output / "model_source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config = yaml.safe_load(args.manifest_config.read_text(encoding="utf-8"))
    checkpoint_manifest = build_manifest(
        model_root=output,
        config=config,
        source_manifest=source_manifest,
    )
    checkpoint_manifest_path = output / "checkpoint_manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                **source_manifest,
                "checkpoint_manifest": str(checkpoint_manifest_path),
                "manifest_status": checkpoint_manifest["status"],
                "total_bytes": checkpoint_manifest["summary"]["total_bytes"],
                "submission_bytes": checkpoint_manifest["summary"]["submission_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if checkpoint_manifest["status"] != "pass" and not args.allow_manifest_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
