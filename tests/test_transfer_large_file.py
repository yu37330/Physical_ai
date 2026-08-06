from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/transfer_large_file.py"
PART_MIB = 1


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source" / "submission.zip"
    path.parent.mkdir()
    # Two and a half parts, so the last one is short.
    path.write_bytes(bytes(range(256)) * (10 * 1024))
    return path


def run(command: str, source: Path, staging: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), command,
         "--source", str(source), "--staging", str(staging),
         "--part-mib", str(kwargs.pop("part_mib", PART_MIB))],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, **kwargs,
    )


def drain(source: Path, staging: Path) -> list[bytes]:
    """Carve, collect, and acknowledge every part, as the operator would."""
    collected = []
    while True:
        assert run("next", source, staging).returncode == 0
        staged = [p for p in staging.glob("submission.zip.part.*")]
        if not staged:
            break
        collected.append(staged[0].read_bytes())
        assert run("done", source, staging).returncode == 0
    return collected


def test_the_parts_rejoin_into_the_original_file(source: Path, tmp_path: Path) -> None:
    """The whole point: what lands on the other machine has to be byte-identical,
    because the archive is checked against the SHA256 recorded at build time."""
    parts = drain(source, tmp_path / "staging")

    rejoined = b"".join(parts)
    assert rejoined == source.read_bytes()
    assert hashlib.sha256(rejoined).hexdigest() == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_part_names_sort_in_transfer_order(source: Path, tmp_path: Path) -> None:
    """Windows rejoins with Sort-Object Name, so part.10 must not sort before
    part.2 -- that reassembles the archive silently out of order."""
    staging = tmp_path / "staging"
    drain(source, staging)

    state = json.loads((staging / "transfer_state.json").read_text(encoding="utf-8"))
    names = [state["parts"][str(i)]["name"] for i in range(state["part_count"])]
    assert names == sorted(names)


def test_a_staged_part_is_not_recarved(source: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    assert run("next", source, staging).returncode == 0
    staged = next(staging.glob("submission.zip.part.*"))
    staged.write_bytes(b"downloaded already")

    result = run("next", source, staging)

    assert "Already staged" in result.stdout
    assert staged.read_bytes() == b"downloaded already"


def test_changing_the_part_size_midway_is_refused(source: Path, tmp_path: Path) -> None:
    """Parts carved at two different sizes overlap or leave a hole, and the
    rejoined file would fail its checksum with nothing to point at."""
    staging = tmp_path / "staging"
    assert run("next", source, staging).returncode == 0
    assert run("done", source, staging).returncode == 0

    result = run("next", source, staging, part_mib=2)

    assert result.returncode != 0
    assert "Part size changed" in result.stderr


def test_a_rebuilt_source_is_refused(source: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    assert run("next", source, staging).returncode == 0
    assert run("done", source, staging).returncode == 0
    source.write_bytes(source.read_bytes() + b"rebuilt")

    result = run("next", source, staging)

    assert result.returncode != 0
    assert "changed size" in result.stderr


def test_status_reports_progress_without_carving(source: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    assert run("next", source, staging).returncode == 0
    assert run("done", source, staging).returncode == 0

    result = run("status", source, staging)

    assert "1/3 parts transferred" in result.stdout
    assert not list(staging.glob("submission.zip.part.*"))


def test_staging_inside_the_source_directory_is_refused(
    source: Path, tmp_path: Path
) -> None:
    """Carving into the same directory would put the parts next to a file the
    caller is about to delete, and doubles the space needed there."""
    result = run("next", source, source.parent)

    assert result.returncode != 0
    assert "Staging directory" in result.stderr
