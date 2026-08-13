from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/patch_submission_zip.py"

WEIGHT_BYTES = b"\x00\x01\x02\x03" * 64
ORIGINAL_REQUIREMENTS = "torch==2.2.0\nnumpy==1.26.4\n"
FIXED_REQUIREMENTS = "torch==2.2.0\nnvidia-nvjitlink-cu12==12.1.105\nnumpy==1.26.4\n"


def _build_archive(path: Path, *, prefix: str = "") -> None:
    """`build_submission_zip.py`と同じ格納方法で最小の提出Archiveを作る。"""
    # Weights stored, everything else deflated, matching build_submission_zip.py.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr(f"{prefix}policy_server.py", "print('policy')\n")
        archive.writestr(f"{prefix}requirements.txt", ORIGINAL_REQUIREMENTS)
        archive.writestr(
            f"{prefix}model.safetensors",
            WEIGHT_BYTES,
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr(f"{prefix}prismatic/__init__.py", "")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def replacement(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(FIXED_REQUIREMENTS, encoding="utf-8")
    return path


def test_only_the_requirements_entry_changes(tmp_path: Path, replacement: Path) -> None:
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    completed = _run(
        ["--input", str(source), "--output", str(output), "--requirements", str(replacement)]
    )
    assert completed.returncode == 0, completed.stderr

    with zipfile.ZipFile(output) as archive:
        assert archive.read("requirements.txt").decode() == FIXED_REQUIREMENTS
        assert archive.read("policy_server.py").decode() == "print('policy')\n"
        assert archive.read("model.safetensors") == WEIGHT_BYTES
        assert sorted(archive.namelist()) == [
            "model.safetensors",
            "policy_server.py",
            "prismatic/__init__.py",
            "requirements.txt",
        ]


def test_stored_weights_are_not_recompressed(tmp_path: Path, replacement: Path) -> None:
    """重みをDEFLATEDへ変えると、14GBのArchiveで書き出しが数十分に伸びる。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    _run(["--input", str(source), "--output", str(output), "--requirements", str(replacement)])

    with zipfile.ZipFile(output) as archive:
        assert archive.getinfo("model.safetensors").compress_type == zipfile.ZIP_STORED
        assert archive.getinfo("policy_server.py").compress_type == zipfile.ZIP_DEFLATED


def test_a_single_top_folder_layout_is_handled(tmp_path: Path, replacement: Path) -> None:
    """運営Validatorが受け付けるもう一方のレイアウトも直せる。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source, prefix="my_submission/")

    completed = _run(
        ["--input", str(source), "--output", str(output), "--requirements", str(replacement)]
    )
    assert completed.returncode == 0, completed.stderr

    with zipfile.ZipFile(output) as archive:
        assert archive.read("my_submission/requirements.txt").decode() == FIXED_REQUIREMENTS
    report = json.loads((tmp_path / "submission_patched.zip.json").read_text(encoding="utf-8"))
    assert report["replaced_entry"] == "my_submission/requirements.txt"


def test_the_source_archive_is_left_untouched(tmp_path: Path, replacement: Path) -> None:
    """差し替えに失敗しても提出可能な現物が残っていることが、この経路の前提。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)
    before = source.read_bytes()

    _run(["--input", str(source), "--output", str(output), "--requirements", str(replacement)])

    assert source.read_bytes() == before


def test_writing_over_the_source_is_refused(tmp_path: Path, replacement: Path) -> None:
    source = tmp_path / "submission.zip"
    _build_archive(source)

    completed = _run(
        ["--input", str(source), "--output", str(source), "--requirements", str(replacement)]
    )
    assert completed.returncode != 0
    assert "must differ" in completed.stderr


def test_an_archive_without_requirements_is_rejected(tmp_path: Path, replacement: Path) -> None:
    source = tmp_path / "submission.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("policy_server.py", "print('policy')\n")

    completed = _run(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.zip"),
            "--requirements",
            str(replacement),
        ]
    )
    assert completed.returncode != 0
    assert "requirements.txt" in completed.stderr


def test_a_deeply_nested_requirements_is_not_mistaken_for_the_real_one(
    tmp_path: Path, replacement: Path
) -> None:
    """`prismatic/.../requirements.txt`のような同名ファイルは提出物が読むものではない。"""
    source = tmp_path / "submission.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("policy_server.py", "print('policy')\n")
        archive.writestr("vendor/prismatic/extras/requirements.txt", "irrelevant\n")

    completed = _run(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.zip"),
            "--requirements",
            str(replacement),
        ]
    )
    assert completed.returncode != 0


def _load_module():
    spec = importlib.util.spec_from_file_location("patch_submission_zip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entries_past_the_zip64_threshold_are_written(
    tmp_path: Path, replacement: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 7B checkpoint shard is well past 4GB.

    zipfile decides whether to write the ZIP64 extension from the size the
    ZipInfo declares *before* the data is written, so an entry built without
    `file_size` aborts with "File size too large, try using force_zip64" -- and
    only once it reaches the first oversized member, which on a real submission
    is several minutes into a 14GB copy. Driving the threshold down reproduces
    that on bytes a test can afford.
    """
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 64)
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)
    assert source.stat().st_size > 64

    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--requirements",
            str(replacement),
        ],
    )
    module.main()

    with zipfile.ZipFile(output) as archive:
        assert archive.read("model.safetensors") == WEIGHT_BYTES
        assert archive.read("requirements.txt").decode() == FIXED_REQUIREMENTS


def test_a_failed_patch_leaves_no_partial_archive(
    tmp_path: Path, replacement: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """14GB of half-written archive is dead weight on the disk that has to hold
    the next attempt."""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    module = _load_module()
    monkeypatch.setattr(
        module.shutil, "copyfileobj", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--requirements",
            str(replacement),
        ],
    )
    with pytest.raises(RuntimeError):
        module.main()

    assert not output.exists()


def test_the_report_records_both_hashes(tmp_path: Path, replacement: Path) -> None:
    """提出記録は差し替え前後の両方を残す。どのArchiveから作ったかが後で要る。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    completed = _run(
        ["--input", str(source), "--output", str(output), "--requirements", str(replacement)]
    )
    report = json.loads(completed.stdout)

    assert report["replaced_entry"] == "requirements.txt"
    assert report["input_sha256"] != report["sha256"]
    assert len(report["sha256"]) == 64
    assert report["entries"] == report["source_entries"]
    assert report["zip_bytes"] == output.stat().st_size
