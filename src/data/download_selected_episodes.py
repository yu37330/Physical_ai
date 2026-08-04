from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only files listed in an episode download plan")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download selected episodes") from exc

    from .hf_download import download_files_verified

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source = plan["source"]
    # 800 episodes is roughly 2,400 files, which trips Hugging Face's anonymous
    # rate limit partway through and can leave the snapshot incomplete without
    # raising. The plan lists exact paths, so confirm every one arrived.
    verification = download_files_verified(
        repo_id=source["repo_id"],
        repo_type="dataset",
        revision=source["resolved_revision"],
        local_dir=args.output_dir,
        relative_paths=plan["patterns"],
    )
    record = {
        "repo_id": source["repo_id"],
        "resolved_revision": source["resolved_revision"],
        "episode_count": plan["episode_count"],
        "include_videos": plan["include_videos"],
        "local_dir": str(args.output_dir.resolve()),
        "plan": str(args.plan),
        "verification": verification,
    }
    (args.output_dir / "selected_download_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
