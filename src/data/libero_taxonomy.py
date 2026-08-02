from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

SUITES = ("spatial", "object", "goal", "long")


def normalize_instruction(text: str) -> str:
    """Normalize an instruction for stable exact matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_suite_map(path: str | Path) -> dict[str, str]:
    """Return normalized instruction -> suite and reject duplicates."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    suites = payload.get("suites", {})
    result: dict[str, str] = {}
    for suite in SUITES:
        tasks = suites.get(suite)
        if not isinstance(tasks, list):
            raise ValueError(f"Missing task list for suite: {suite}")
        for task in tasks:
            normalized = normalize_instruction(str(task))
            previous = result.get(normalized)
            if previous is not None:
                raise ValueError(f"Task appears in multiple suites: {task!r}: {previous}, {suite}")
            result[normalized] = suite
    if len(result) != 40:
        raise ValueError(f"Expected 40 unique tasks, found {len(result)}")
    return result


def classify_instruction(instruction: str, suite_map: Mapping[str, str]) -> str:
    """Classify one canonical LIBERO instruction, returning 'unknown' if absent."""
    return suite_map.get(normalize_instruction(instruction), "unknown")
