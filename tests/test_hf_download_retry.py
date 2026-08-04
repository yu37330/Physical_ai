from __future__ import annotations

import sys
import types

import pytest

from src.data import hf_download


class _Response:
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after else {}


class _HfHubHTTPError(Exception):
    def __init__(self, message: str, response: _Response | None = None) -> None:
        super().__init__(message)
        self.response = response


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_download.time, "sleep", lambda seconds: None)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch):
    """Stand in for huggingface_hub, which snapshot_with_retry imports lazily."""
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(dict(kwargs))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    outcomes: list = []
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = _HfHubHTTPError
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setenv("HF_TOKEN", "set-so-the-warning-is-quiet")
    return types.SimpleNamespace(calls=calls, outcomes=outcomes)


def test_a_successful_download_is_not_retried(hub) -> None:
    hub.outcomes.append("/local/path")

    assert hf_download.snapshot_with_retry(repo_id="x") == "/local/path"
    assert len(hub.calls) == 1


def test_rate_limiting_is_retried_and_drops_to_one_worker(hub) -> None:
    """Concurrency is what trips the limit, so a retry must stop competing with
    itself rather than repeating the same burst."""
    hub.outcomes.extend(
        [
            _HfHubHTTPError("429 Client Error: Too Many Requests", _Response(429)),
            "/local/path",
        ]
    )

    assert hf_download.snapshot_with_retry(repo_id="x") == "/local/path"
    assert len(hub.calls) == 2
    assert "max_workers" not in hub.calls[0]
    assert hub.calls[1]["max_workers"] == 1


def test_other_http_errors_are_not_retried(hub) -> None:
    """A 404 will never succeed on retry; failing immediately keeps the message."""
    hub.outcomes.append(_HfHubHTTPError("404 Client Error: Not Found", _Response(404)))

    with pytest.raises(_HfHubHTTPError, match="404"):
        hf_download.snapshot_with_retry(repo_id="x")
    assert len(hub.calls) == 1


def test_persistent_rate_limiting_says_to_set_a_token(hub) -> None:
    hub.outcomes.extend(
        _HfHubHTTPError("429 Too Many Requests", _Response(429))
        for _ in range(hf_download.MAX_ATTEMPTS)
    )

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hf_download.snapshot_with_retry(repo_id="x")
    assert len(hub.calls) == hf_download.MAX_ATTEMPTS


def test_retry_after_header_is_honoured() -> None:
    error = _HfHubHTTPError("429", _Response(429, retry_after="30"))

    assert hf_download._retry_delay(error, attempt=0) == pytest.approx(31.0)


def test_backoff_grows_and_is_capped() -> None:
    error = _HfHubHTTPError("429", _Response(429))

    first = hf_download._retry_delay(error, attempt=0)
    later = hf_download._retry_delay(error, attempt=1)

    assert first < later
    assert hf_download._retry_delay(error, attempt=10) <= hf_download.MAX_RETRY_SECONDS


@pytest.mark.parametrize(
    "variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
)
def test_either_token_variable_counts_as_configured(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_download.token_is_configured() is False

    monkeypatch.setenv(variable, "value")
    assert hf_download.token_is_configured() is True
