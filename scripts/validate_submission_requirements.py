from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROHIBITED_PATTERNS = {
    "git_dependency": re.compile(r"(?:git\+|git:)", re.IGNORECASE),
    "remote_url": re.compile(r"https?://", re.IGNORECASE),
    "editable_install": re.compile(r"^\s*-e(?:\s|$)", re.MULTILINE),
    "custom_index": re.compile(r"^\s*--(?:extra-index-url|index-url|find-links)\b", re.MULTILINE),
    "local_path": re.compile(r"^\s*(?:\.{1,2}/|/|file:)", re.MULTILINE),
}


def validate_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    violations: list[dict[str, str]] = []
    for name, pattern in PROHIBITED_PATTERNS.items():
        match = pattern.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            violations.append({"file": str(path), "rule": name, "line": str(line)})
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PARC submission requirements files")
    parser.add_argument("root", type=Path, nargs="?", default=Path("submission"))
    args = parser.parse_args()

    files = sorted(args.root.glob("**/requirements.txt"))
    if not files:
        raise SystemExit(f"No requirements.txt found under {args.root}")
    violations = [violation for path in files for violation in validate_file(path)]
    payload = {
        "status": "pass" if not violations else "fail",
        "files": [str(path) for path in files],
        "violations": violations,
    }
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
