from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_patterns(episodes: list[dict[str, Any]], include_videos: bool) -> list[str]:
    patterns: set[str] = set()
    for row in episodes:
        episode_index = int(row["episode_index"])
        chunk = episode_index // 1000
        patterns.add(f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")
        if include_videos:
            for key in ("observation.images.front", "observation.images.wrist"):
                patterns.add(f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4")
    return sorted(patterns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build episode-selective Hugging Face allow_patterns")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    patterns = build_patterns(selection["episodes"], args.include_videos)
    payload = {
        "source": selection["source"],
        "include_videos": args.include_videos,
        "episode_count": len(selection["episodes"]),
        "patterns": patterns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episode_count": payload["episode_count"], "file_count": len(patterns)}, indent=2))


if __name__ == "__main__":
    main()
