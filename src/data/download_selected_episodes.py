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
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download selected episodes") from exc

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source = plan["source"]
    local = snapshot_download(
        repo_id=source["repo_id"],
        repo_type="dataset",
        revision=source["resolved_revision"],
        local_dir=args.output_dir,
        allow_patterns=plan["patterns"],
    )
    record = {
        "repo_id": source["repo_id"],
        "resolved_revision": source["resolved_revision"],
        "episode_count": plan["episode_count"],
        "include_videos": plan["include_videos"],
        "local_dir": str(Path(local).resolve()),
        "plan": str(args.plan),
    }
    (args.output_dir / "selected_download_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
