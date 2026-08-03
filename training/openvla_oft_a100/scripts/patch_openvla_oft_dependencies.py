"""Make the pinned OpenVLA-OFT dependency set installable on Python 3.12.

Current Colab runtimes are Python 3.12, where the pinned OpenVLA-OFT commit
cannot be installed at all:

- `tensorflow==2.15.0` and `tensorflow_datasets==4.9.3` ship no 3.12 wheels.
- `tensorflow_graphics==2021.12.3` requires `tensorflow-addons`, whose final
  release (0.23.0) publishes neither a 3.12 wheel nor an sdist.

This script relaxes the first two pins to the same TensorFlow line that
`requirements-data.txt` already selects for 3.12, and drops the
`tensorflow_graphics` entry so pip stops resolving `tensorflow-addons`.
`bootstrap_colab.sh` then reinstalls `tensorflow_graphics` with `--no-deps`:
`prismatic` only reaches `tensorflow_graphics.geometry.transformation`, whose
import chain needs nothing beyond TensorFlow. The heavier `nn`/`rendering`
subpackages that pull `tensorflow-addons` stay behind tfg's docs-only guard.

On Python 3.10/3.11 this script is a no-op so the originally tested OpenVLA-OFT
dependency set stays reproducible.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "PARC_PY312_TENSORFLOW_PATCH"

# Package name -> replacement specifier, or None to drop the dependency entirely.
PY312_REPLACEMENTS: dict[str, str | None] = {
    "tensorflow": "tensorflow>=2.19,<2.20",
    "tensorflow_datasets": "tensorflow_datasets>=4.9.9,<4.10",
    "tensorflow_graphics": None,
}


def patch(path: Path, python_version: tuple[int, int]) -> dict[str, str | None]:
    """Rewrite TensorFlow pins in `path`. Returns the applied replacements."""
    source = path.read_text(encoding="utf-8")
    if python_version < (3, 12) or MARKER in source:
        return {}

    applied: dict[str, str | None] = {}
    for package, replacement in PY312_REPLACEMENTS.items():
        # Matches a dependency entry such as `    "tensorflow==2.15.0",` and, when
        # the entry is dropped, its whole line so no stray comma is left behind.
        entry = rf'"{re.escape(package)}\s*==[^"]*"'
        pattern = re.compile(rf"^[ \t]*{entry},?[ \t]*\r?\n" if replacement is None else entry, re.M)
        match = pattern.search(source)
        if match is None:
            raise RuntimeError(
                f"Could not locate a pinned {package} dependency in {path}; "
                "the pinned OpenVLA-OFT source changed"
            )
        applied[re.search(entry, match.group(0)).group(0).strip('"')] = replacement
        source = pattern.sub("" if replacement is None else f'"{replacement}"', source, count=1)

    source += f"\n# {MARKER}: TensorFlow dependencies adjusted for Python 3.12 runtimes.\n"
    path.write_text(source, encoding="utf-8")
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", type=Path)
    args = parser.parse_args()

    applied = patch(args.pyproject, sys.version_info[:2])
    if not applied:
        print(f"No dependency patch needed for Python {sys.version_info[0]}.{sys.version_info[1]}")
        return
    for original, replacement in applied.items():
        print(f"Patched: {original} -> {replacement or 'removed'}")


if __name__ == "__main__":
    main()
