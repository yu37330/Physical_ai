from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only LeRobot metadata at a fixed Hugging Face revision")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("lerobot_v2_1", "lerobot_v3"), required=True)
    args = parser.parse_args()

    # Imported first: it sets HF_HUB_DOWNLOAD_TIMEOUT, which huggingface_hub only
    # reads at its own import time.
    from .hf_download import snapshot_with_retry

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download dataset metadata") from exc

    patterns = ["meta/info.json"]
    if args.format == "lerobot_v2_1":
        patterns += ["meta/tasks.jsonl", "meta/episodes.jsonl", "meta/episodes_stats.jsonl", "norm_stats.json"]
    else:
        patterns += ["meta/tasks.parquet", "meta/episodes/**/*.parquet", "meta/stats.json"]

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision, files_metadata=False)
    resolved_revision = info.sha
    local = snapshot_with_retry(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=args.output_dir,
        allow_patterns=patterns,
    )
    record = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "format": args.format,
        "local_dir": str(Path(local).resolve()),
        "allow_patterns": patterns,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata_download_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
