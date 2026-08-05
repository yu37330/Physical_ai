"""Post a short run notification to a Discord webhook.

Stage A and the submission build run for half an hour or more, and Colab
reclaims an idle runtime out from under them, so knowing the moment one finishes
is worth something.

The webhook URL is a bearer credential -- anyone holding it can post to the
channel -- so it is never an argument and never lands in the repository. It comes
from DISCORD_WEBHOOK_URL, or from a file on Drive:

    /content/drive/MyDrive/PARC2026/00_admin/discord_webhook.txt

Nothing here raises on a delivery failure. A notification that cannot be sent is
not a reason to fail a run that just succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_WEBHOOK_FILE = Path("/content/drive/MyDrive/PARC2026/00_admin/discord_webhook.txt")
TIMEOUT_SECONDS = 15


def resolve_webhook(webhook_file: Path) -> str | None:
    from_env = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if from_env:
        return from_env
    if webhook_file.is_file():
        value = webhook_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def format_message(message: str, elapsed_seconds: int | None) -> str:
    parts = [message]
    if elapsed_seconds is not None:
        minutes, seconds = divmod(max(elapsed_seconds, 0), 60)
        hours, minutes = divmod(minutes, 60)
        parts.append(
            f"({hours}h{minutes:02d}m)" if hours else f"({minutes}m{seconds:02d}s)"
        )
    parts.append(f"[{socket.gethostname()}]")
    return " ".join(parts)


def send(webhook: str, content: str) -> bool:
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"content": content}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "parc2026-notify"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        # Deliberately not the exception text: a redirect or proxy error can echo
        # the request URL, and the URL is the credential.
        print(f"Discord notification failed: {type(error).__name__}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--elapsed-seconds", type=int)
    parser.add_argument("--webhook-file", type=Path, default=DEFAULT_WEBHOOK_FILE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message instead of posting it. Never prints the webhook.",
    )
    args = parser.parse_args()

    content = format_message(args.message, args.elapsed_seconds)
    if args.dry_run:
        print(content)
        return

    webhook = resolve_webhook(args.webhook_file)
    if not webhook:
        print(
            "No Discord webhook configured; skipping the notification.\n"
            f"Set DISCORD_WEBHOOK_URL or write it to {args.webhook_file}.",
            file=sys.stderr,
        )
        return

    if send(webhook, content):
        print("Notified Discord.")


if __name__ == "__main__":
    main()
