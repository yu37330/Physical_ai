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


def token_is_configured() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


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
        except HfHubHTTPError as error:
            if not _is_rate_limit(error):
                raise
            last_error = error
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = _retry_delay(error, attempt)
            print(
                f"Rate limited by Hugging Face; retrying in {delay:.0f}s "
                f"(attempt {attempt + 2}/{MAX_ATTEMPTS}). Completed files are kept.",
                flush=True,
            )
            time.sleep(delay)
            if serialize_on_retry:
                # Concurrency is what trips the limit; stop competing with ourselves.
                kwargs["max_workers"] = 1

    raise RuntimeError(
        "Hugging Face kept rate limiting the download. Set HF_TOKEN to raise the "
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
    }


def resolve_local_dir(path: Path) -> Path:
    return Path(path).resolve()
