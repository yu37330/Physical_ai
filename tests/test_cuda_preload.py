from __future__ import annotations

import sys
from pathlib import Path

import pytest

from submission.openvla_oft_offline.runtime import cuda_preload


def _plant(root: Path, package: str, filename: str) -> Path:
    lib = root / "nvidia" / package / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    target = lib / filename
    target.write_bytes(b"")
    return target


def test_it_is_found_in_the_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv = tmp_path / "venv/lib/python3.10/site-packages"
    planted = _plant(venv, "nvjitlink", cuda_preload.SONAME)
    monkeypatch.setattr(sys, "path", [str(venv)])

    assert cuda_preload.find_nvjitlink() == str(planted)


def test_it_is_found_in_the_image_when_the_venv_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the scoring venv is built with --system-site-packages, so
    the wheel may land on either side and the submission cannot control which."""
    venv = tmp_path / "venv/lib/python3.10/site-packages"
    system = tmp_path / "usr/lib/python3/dist-packages"
    _plant(venv, "cusparse", "libcusparse.so.12")
    planted = _plant(system, "nvjitlink", cuda_preload.SONAME)
    monkeypatch.setattr(sys, "path", [str(venv), str(system)])

    assert cuda_preload.find_nvjitlink() == str(planted)


def test_the_cuda_13_library_is_not_mistaken_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image ships libnvJitLink.so.13 in the same directory. It does not
    satisfy a DT_NEEDED on .so.12, so matching it would report success and leave
    the server to fail exactly as before."""
    system = tmp_path / "usr/lib/python3/dist-packages"
    _plant(system, "nvjitlink", "libnvJitLink.so.13")
    monkeypatch.setattr(sys, "path", [str(system)])

    assert cuda_preload.find_nvjitlink() is None


def test_a_versioned_filename_still_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "site-packages"
    planted = _plant(system, "nvjitlink", cuda_preload.SONAME + ".1.105")
    monkeypatch.setattr(sys, "path", [str(system)])

    assert cuda_preload.find_nvjitlink() == str(planted)


def test_missing_everywhere_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preloading is unnecessary where the loader already resolves it, so absence
    must not break a working environment."""
    empty = tmp_path / "site-packages"
    empty.mkdir()
    monkeypatch.setattr(sys, "path", [str(empty)])

    assert cuda_preload.preload_nvjitlink() is None
    assert cuda_preload.SONAME in capsys.readouterr().err


def test_the_diagnostic_names_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed preload has to leave enough in the scoring log to tell which
    site-packages received which wheels."""
    venv = tmp_path / "venv/lib/python3.10/site-packages"
    system = tmp_path / "usr/lib/python3/dist-packages"
    _plant(venv, "cusparse", "libcusparse.so.12")
    _plant(system, "nvjitlink", "libnvJitLink.so.13")
    monkeypatch.setattr(sys, "path", [str(venv), str(system)])

    cuda_preload.preload_nvjitlink()

    err = capsys.readouterr().err
    assert "cusparse" in err
    assert "libnvJitLink.so.13" in err


def test_configure_offline_environment_preloads_before_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The call has to sit in configure_offline_environment: the runtime invokes
    it first, and the next thing that happens is the transformers import that
    pulls in torch."""
    from submission.openvla_oft_offline.runtime import offline_env

    calls: list[str] = []
    monkeypatch.setattr(offline_env, "preload_nvjitlink", lambda: calls.append("preload"))

    offline_env.configure_offline_environment()

    assert calls == ["preload"]
