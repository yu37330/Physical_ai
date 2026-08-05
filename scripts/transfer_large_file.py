"""Move a file off Colab in parts, one at a time.

The Jupyter contents API that VS Code downloads through base64-encodes the whole
file into a JSON response, so a 14GB submission archive fails with a 500 before a
byte reaches the disk. Splitting it up front would need a second copy of the
archive, and /content has nowhere near that much room, so each part is carved out
on demand and deleted once it has landed.

    python scripts/transfer_large_file.py next   --source ZIP --staging DIR
    python scripts/transfer_large_file.py done   --source ZIP --staging DIR
    python scripts/transfer_large_file.py status --source ZIP --staging DIR

Rejoining on Windows, once every part is downloaded:

    $out = [IO.File]::Create("$HOME\\Downloads\\submission.zip")
    Get-ChildItem "$HOME\\Downloads\\parts\\*.part.*" | Sort-Object Name | ForEach-Object {
        $in = [IO.File]::OpenRead($_.FullName); $in.CopyTo($out); $in.Close()
    }
    $out.Close()

Sort-Object Name is why parts are numbered with a fixed width: part.10 must not
sort before part.2, or the archive is silently reassembled out of order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

CHUNK = 16 * 1024 * 1024
STATE_NAME = "transfer_state.json"


def state_path(staging: Path) -> Path:
    return staging / STATE_NAME


def load_state(source: Path, staging: Path, part_mib: int) -> dict:
    path = state_path(staging)
    size = source.stat().st_size
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        # Both of these silently corrupt the rejoined file rather than failing,
        # so they are worth refusing over.
        if state["source_size"] != size:
            raise SystemExit(
                f"{source.name} changed size since the transfer started "
                f"({state['source_size']} -> {size}). Start over with a clean staging "
                f"directory."
            )
        if state["part_bytes"] != part_mib * 1024**2:
            raise SystemExit(
                f"Part size changed mid-transfer "
                f"({state['part_bytes'] // 1024**2} MiB -> {part_mib} MiB). "
                f"Re-run with --part-mib {state['part_bytes'] // 1024**2}."
            )
        return state

    part_bytes = part_mib * 1024**2
    state = {
        "source": str(source),
        "source_size": size,
        "part_bytes": part_bytes,
        "part_count": (size + part_bytes - 1) // part_bytes,
        "completed": [],
        "parts": {},
    }
    staging.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def save_state(staging: Path, state: dict) -> None:
    state_path(staging).write_text(json.dumps(state, indent=2), encoding="utf-8")


def part_name(source: Path, index: int, count: int) -> str:
    width = max(3, len(str(count - 1)))
    return f"{source.name}.part.{index:0{width}d}"


def staged_parts(source: Path, staging: Path) -> list[Path]:
    return sorted(p for p in staging.glob(f"{source.name}.part.*") if p.is_file())


def carve(source: Path, target: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with source.open("rb") as reader, target.open("wb") as writer:
        reader.seek(offset)
        while remaining:
            block = reader.read(min(CHUNK, remaining))
            if not block:
                break
            writer.write(block)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def cmd_next(source: Path, staging: Path, state: dict) -> None:
    existing = staged_parts(source, staging)
    if existing:
        print(f"Already staged: {existing[0]}")
        print("Download it, then run 'done' to free the space and advance.")
        return

    index = len(state["completed"])
    if index >= state["part_count"]:
        print("Every part has been transferred.")
        cmd_status(source, staging, state)
        return

    offset = index * state["part_bytes"]
    length = min(state["part_bytes"], state["source_size"] - offset)
    free = shutil.disk_usage(staging).free
    if free < length + 128 * 1024**2:
        raise SystemExit(
            f"Not enough space to carve part {index}: needs "
            f"{length / 1024**2:.0f} MiB, {free / 1024**2:.0f} MiB free at {staging}."
        )

    target = staging / part_name(source, index, state["part_count"])
    sha = carve(source, target, offset, length)
    state["parts"][str(index)] = {"name": target.name, "bytes": length, "sha256": sha}
    save_state(staging, state)

    print(f"Part {index + 1} of {state['part_count']}: {target}")
    print(f"  {length / 1024**2:.0f} MiB  sha256 {sha}")
    print("Download it in VS Code, then run 'done'.")


def cmd_done(source: Path, staging: Path, state: dict) -> None:
    existing = staged_parts(source, staging)
    if not existing:
        print("Nothing staged. Run 'next' to carve the next part.")
        return

    index = len(state["completed"])
    recorded = state["parts"].get(str(index))
    for path in existing:
        if recorded and path.name != recorded["name"]:
            raise SystemExit(f"Unexpected file in staging: {path}")
        path.unlink()
    state["completed"].append(index)
    save_state(staging, state)
    cmd_status(source, staging, state)


def cmd_status(source: Path, staging: Path, state: dict) -> None:
    done = len(state["completed"])
    total = state["part_count"]
    moved = sum(state["parts"][str(i)]["bytes"] for i in state["completed"])
    print(f"{done}/{total} parts transferred ({moved / 1024**3:.2f} GiB of "
          f"{state['source_size'] / 1024**3:.2f} GiB)")
    if done == total:
        print("\nRejoin on Windows, then check the SHA256 against the build record.")
        for index in state["completed"]:
            entry = state["parts"][str(index)]
            print(f"  {entry['name']}  {entry['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["next", "done", "status"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=1400)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"Source file not found: {source}")
    staging = args.staging.resolve()
    if staging == source.parent:
        raise SystemExit("Staging directory must not be the source's own directory")

    state = load_state(source, staging, args.part_mib)
    {"next": cmd_next, "done": cmd_done, "status": cmd_status}[args.command](
        source, staging, state
    )


if __name__ == "__main__":
    main()
