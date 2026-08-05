from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/notify_discord.py"
SECRET = "https://discord.com/api/webhooks/000/AAAAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("notify_discord", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, env=env,
    )


def test_a_missing_webhook_does_not_fail_the_run(tmp_path: Path) -> None:
    """The notification is the last thing a half-hour training run does. Exiting
    non-zero here would report a successful run as a failure."""
    result = run(["--message", "Stage A s2", "--webhook-file", str(tmp_path / "absent")])

    assert result.returncode == 0
    assert "No Discord webhook configured" in result.stderr


def test_the_webhook_is_never_printed(tmp_path: Path) -> None:
    webhook_file = tmp_path / "discord_webhook.txt"
    webhook_file.write_text(SECRET, encoding="utf-8")

    result = run([
        "--message", "Stage A s2", "--webhook-file", str(webhook_file), "--dry-run",
    ])

    assert result.returncode == 0
    assert SECRET not in result.stdout
    assert SECRET not in result.stderr


def test_the_environment_wins_over_the_drive_file(module, tmp_path: Path) -> None:
    """A run can point somewhere else without editing the file every wrapper reads."""
    webhook_file = tmp_path / "discord_webhook.txt"
    webhook_file.write_text("from-file", encoding="utf-8")

    import os
    os.environ["DISCORD_WEBHOOK_URL"] = "from-env"
    try:
        assert module.resolve_webhook(webhook_file) == "from-env"
        del os.environ["DISCORD_WEBHOOK_URL"]
        assert module.resolve_webhook(webhook_file) == "from-file"
    finally:
        os.environ.pop("DISCORD_WEBHOOK_URL", None)


def test_a_blank_webhook_file_counts_as_unconfigured(module, tmp_path: Path) -> None:
    webhook_file = tmp_path / "discord_webhook.txt"
    webhook_file.write_text("\n  \n", encoding="utf-8")

    assert module.resolve_webhook(webhook_file) is None


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(0, "(0m00s)"), (95, "(1m35s)"), (3600, "(1h00m)"), (5430, "(1h30m)")],
)
def test_elapsed_time_is_readable(module, elapsed: int, expected: str) -> None:
    assert expected in module.format_message("Stage A s2", elapsed)
