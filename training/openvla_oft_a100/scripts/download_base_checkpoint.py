from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = HfApi().model_info(args.repo_id, revision=args.revision)
    resolved_sha = info.sha
    output = args.output.expanduser().resolve()
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved_sha,
        local_dir=output,
        local_dir_use_symlinks=False,
    )
    manifest = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_sha,
    }
    (output / "model_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
