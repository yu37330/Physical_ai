"""Hugging Face downloads that survive rate limiting.

Fetching 800 episodes means about 2,400 files, and anonymous requests from a
Colab IP hit HF's rate limit partway through:

    429 Too Many Requests ... We had to rate limit your IP

The competition's own reference notebook carries the same retry helper, so this
is expected rather than exceptional. Two things help, and they compose:

- Set HF_TOKEN. Authenticated requests get a far higher limit, and this is what
  the 429 body itself asks for. Without it a large download will keep stalling.
- Back off and retry on 429, honouring Retry-After, and fall back to a single
  worker so a resumed download stops hammering the API.

Already-downloaded files are skipped on resume, so a retried snapshot picks up
where it stopped rather than starting over.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

MAX_ATTEMPTS = 6
DEFAULT_RETRY_SECONDS = 15.0
MAX_RETRY_SECONDS = 120.0

# huggingface_hub reads this in constants.py at import time, so it has to be set
# before the hub is imported anywhere. Without it a stalled transfer hangs
# indefinitely instead of raising, which is what left a 2,400 file download
# frozen at 1,994 with the process alive and nothing to retry on.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


# Below this many files an anonymous download finishes before the limit bites,
# which keeps the 9-file mini profile usable without a token.
BULK_DOWNLOAD_FILES = 200

TOKEN_INSTRUCTIONS = (
    "Set HF_TOKEN before a bulk download. In a notebook cell:\n"
    "    import getpass, os\n"
    '    os.environ["HF_TOKEN"] = getpass.getpass("HF token: ")\n'
    "Never paste the token into a cell body; the notebook file is tracked in git.\n"
    "Set ALLOW_ANONYMOUS_DOWNLOAD=1 to proceed anyway."
)


def token_is_configured() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def check_token() -> dict[str, Any]:
    """Report whether a token is set and actually accepted by the Hub.

    A revoked or mistyped token is worse than none: huggingface_hub retries
    internally before surfacing an error, so the download simply goes quiet for
    minutes instead of failing. One whoami call settles it up front.
    """
    if not token_is_configured():
        return {"configured": False, "valid": False, "user": None}

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        info = HfApi().whoami(token=token)
    except (HfHubHTTPError, OSError) as error:
        return {"configured": True, "valid": False, "user": None, "error": str(error)}
    return {"configured": True, "valid": True, "user": info.get("name")}


def require_token_for_bulk(file_count: int) -> dict[str, Any]:
    """Stop before a long download that anonymous access cannot finish."""
    status = check_token()
    if file_count < BULK_DOWNLOAD_FILES or os.environ.get("ALLOW_ANONYMOUS_DOWNLOAD") == "1":
        return status

    if not status["configured"]:
        raise SystemExit(
            f"{file_count} files is more than anonymous access will finish.\n"
            + TOKEN_INSTRUCTIONS
        )
    if not status["valid"]:
        raise SystemExit(
            "HF_TOKEN is set but the Hub rejected it, so the download would stall "
            "rather than fail.\n"
            f"{status.get('error', '')}\n" + TOKEN_INSTRUCTIONS
        )
    return status


def _retry_delay(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return min(MAX_RETRY_SECONDS, float(headers["Retry-After"]) + 1.0)
    except (KeyError, TypeError, ValueError):
        return min(MAX_RETRY_SECONDS, DEFAULT_RETRY_SECONDS * (2**attempt) + random.random())


def _is_rate_limit(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    return "429" in str(error) or "rate limit" in str(error).lower()


def _is_transient(error: Exception) -> bool:
    """Whether resuming is worth a try.

    A 2,400 file download stopped dead at 1,994 with the process alive and no
    output for minutes: a connection hung rather than failed. Nothing was raised,
    so retrying on 429 alone never fired. With HF_HUB_DOWNLOAD_TIMEOUT set, that
    hang surfaces as a timeout, and timeouts and dropped connections are exactly
    the case where resuming works, since completed files are kept.
    """
    if _is_rate_limit(error):
        return True
    name = type(error).__name__.lower()
    if any(term in name for term in ("timeout", "connection", "protocol", "incomplete")):
        return True
    text = str(error).lower()
    return any(term in text for term in ("timed out", "timeout", "connection reset", "connection aborted"))


def _warn_without_token() -> None:
    if not token_is_configured():
        print(
            "HF_TOKEN is not set. Anonymous downloads are rate limited and a large "
            "episode fetch will probably stall; see docs/VSCODE_COLAB_WORKFLOW.md.",
            flush=True,
        )


def _call_with_retry(operation, kwargs: dict[str, Any], *, serialize_on_retry: bool) -> Any:
    from huggingface_hub.errors import HfHubHTTPError

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation(**kwargs)
        except (HfHubHTTPError, OSError) as error:
            # OSError covers requests' timeout and connection errors, which is how
            # a stalled transfer surfaces once HF_HUB_DOWNLOAD_TIMEOUT is set.
            if not _is_transient(error):
                raise
            last_error = error
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = _retry_delay(error, attempt)
            reason = "Rate limited by" if _is_rate_limit(error) else "Transfer stalled against"
            print(
                f"{reason} Hugging Face ({type(error).__name__}); retrying in "
                f"{delay:.0f}s (attempt {attempt + 2}/{MAX_ATTEMPTS}). "
                "Completed files are kept.",
                flush=True,
            )
            time.sleep(delay)
            if serialize_on_retry:
                # Concurrency is what trips the limit; stop competing with ourselves.
                kwargs["max_workers"] = 1

    raise RuntimeError(
        "Hugging Face downloads kept failing. Set HF_TOKEN to raise the rate "
        "limit, then re-run; already downloaded files are reused."
    ) from last_error


def snapshot_with_retry(**kwargs: Any) -> str:
    """`snapshot_download` with backoff on rate limiting."""
    from huggingface_hub import snapshot_download

    _warn_without_token()
    return _call_with_retry(snapshot_download, kwargs, serialize_on_retry=True)


def file_with_retry(**kwargs: Any) -> str:
    """`hf_hub_download` with the same backoff, for repairing individual files."""
    from huggingface_hub import hf_hub_download

    return _call_with_retry(hf_hub_download, kwargs, serialize_on_retry=False)


def missing_files(local_dir: Path, relative_paths: list[str]) -> list[str]:
    root = Path(local_dir)
    return [name for name in relative_paths if not (root / name).is_file()]


def download_files_verified(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    relative_paths: list[str],
    repair_attempts: int = 3,
) -> dict[str, Any]:
    """Download an exact file list and confirm every one of them landed.

    snapshot_download can return normally while some files are absent, which a
    rate-limited 800 episode fetch actually produced. The conversion then failed
    28 minutes in on the first missing video. Verifying the list costs seconds, so
    do it here and re-fetch what is missing rather than discovering it later.
    """
    local_dir = Path(local_dir)
    token_status = require_token_for_bulk(len(relative_paths))
    snapshot_with_retry(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=relative_paths,
    )

    repaired: list[str] = []
    for _ in range(repair_attempts):
        missing = missing_files(local_dir, relative_paths)
        if not missing:
            break
        print(
            f"{len(missing)} of {len(relative_paths)} files are missing after the "
            "snapshot; fetching them individually.",
            flush=True,
        )
        for name in missing:
            file_with_retry(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=name,
                local_dir=local_dir,
            )
            repaired.append(name)

    still_missing = missing_files(local_dir, relative_paths)
    if still_missing:
        raise RuntimeError(
            f"{len(still_missing)} files could not be downloaded, first few: "
            f"{still_missing[:5]}"
        )
    return {
        "requested_files": len(relative_paths),
        "repaired_files": sorted(set(repaired)),
        "verified": True,
        "authenticated": token_status["valid"],
    }


def main() -> None:
    """`python -m src.data.hf_download` answers "is my token set and accepted?"."""
    import json

    status = check_token()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status["valid"]:
        print("\n" + TOKEN_INSTRUCTIONS)
        raise SystemExit(1)


if __name__ == "__main__":
    main()


def resolve_local_dir(path: Path) -> Path:
    return Path(path).resolve()
