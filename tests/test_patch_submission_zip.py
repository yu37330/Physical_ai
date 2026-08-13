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
VENDOR_SOURCE = "# vendored prismatic, prepared on Colab\n"
ORIGINAL_REQUIREMENTS = "torch==2.2.0\nnumpy==1.26.4\n"
FIXED_REQUIREMENTS = "torch==2.2.0\nnvidia-nvjitlink-cu12==12.1.105\nnumpy==1.26.4\n"
ORIGINAL_SERVER = "print('policy')\n"
FIXED_SERVER = "print('policy, fixed')\n"
NEW_MODULE = "def preload_nvjitlink():\n    return None\n"


def _build_archive(path: Path, *, prefix: str = "") -> None:
    """`build_submission_zip.py`と同じ格納方法で最小の提出Archiveを作る。"""
    # Weights stored, everything else deflated, matching build_submission_zip.py.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr(f"{prefix}policy_server.py", ORIGINAL_SERVER)
        archive.writestr(f"{prefix}requirements.txt", ORIGINAL_REQUIREMENTS)
        archive.writestr(f"{prefix}runtime/offline_env.py", "# old\n")
        archive.writestr(
            f"{prefix}model_weights/model.safetensors",
            WEIGHT_BYTES,
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr(f"{prefix}prismatic/__init__.py", VENDOR_SOURCE)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def sync_dir(tmp_path: Path) -> Path:
    """Repo上の`submission/openvla_oft_offline`に相当する差し替え元。"""
    root = tmp_path / "source"
    (root / "runtime").mkdir(parents=True)
    (root / "requirements.txt").write_text(FIXED_REQUIREMENTS, encoding="utf-8")
    (root / "policy_server.py").write_text(FIXED_SERVER, encoding="utf-8")
    (root / "runtime/offline_env.py").write_text("# new\n", encoding="utf-8")
    (root / "runtime/cuda_preload.py").write_text(NEW_MODULE, encoding="utf-8")
    return root


def _patch(source: Path, output: Path, sync_dir: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        ["--input", str(source), "--output", str(output), "--sync-dir", str(sync_dir)]
    )


def test_code_is_replaced_and_weights_are_left_alone(
    tmp_path: Path, sync_dir: Path
) -> None:
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    completed = _patch(source, output, sync_dir)
    assert completed.returncode == 0, completed.stderr

    with zipfile.ZipFile(output) as archive:
        assert archive.read("requirements.txt").decode() == FIXED_REQUIREMENTS
        assert archive.read("policy_server.py").decode() == FIXED_SERVER
        assert archive.read("runtime/offline_env.py").decode() == "# new\n"
        # 重みとVendorはRepoに無い。触らないことがこの経路の前提。
        assert archive.read("model_weights/model.safetensors") == WEIGHT_BYTES
        assert archive.read("prismatic/__init__.py").decode() == VENDOR_SOURCE


def test_a_module_the_archive_never_had_is_added(tmp_path: Path, sync_dir: Path) -> None:
    """修正が新しいModuleとして入ることがある。差し替えだけでは届かない。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    completed = _patch(source, output, sync_dir)
    report = json.loads(completed.stdout)

    with zipfile.ZipFile(output) as archive:
        assert archive.read("runtime/cuda_preload.py").decode() == NEW_MODULE
    assert report["added"] == ["runtime/cuda_preload.py"]
    assert "requirements.txt" in report["replaced"]


def test_stored_weights_are_not_recompressed(tmp_path: Path, sync_dir: Path) -> None:
    """重みをDEFLATEDへ変えると、14GBのArchiveで書き出しが数十分に伸びる。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    _patch(source, output, sync_dir)

    with zipfile.ZipFile(output) as archive:
        assert archive.getinfo("model_weights/model.safetensors").compress_type == zipfile.ZIP_STORED
        assert archive.getinfo("policy_server.py").compress_type == zipfile.ZIP_DEFLATED


def test_weights_and_vendor_in_the_sync_dir_are_ignored(
    tmp_path: Path, sync_dir: Path
) -> None:
    """Colabでは`submission/openvla_oft_offline`の下に重みとVendorが実在する。
    そこから同期し始めると、14GBを読み直したうえに壊しかねない。"""
    (sync_dir / "model_weights").mkdir()
    (sync_dir / "model_weights/model.safetensors").write_bytes(b"wrong")
    (sync_dir / "prismatic").mkdir()
    (sync_dir / "prismatic/__init__.py").write_text("# wrong\n", encoding="utf-8")
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    _patch(source, output, sync_dir)

    with zipfile.ZipFile(output) as archive:
        assert archive.read("model_weights/model.safetensors") == WEIGHT_BYTES
        assert archive.read("prismatic/__init__.py").decode() == VENDOR_SOURCE


def test_a_single_top_folder_layout_is_handled(tmp_path: Path, sync_dir: Path) -> None:
    """運営Validatorが受け付けるもう一方のレイアウトも直せる。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source, prefix="my_submission/")

    completed = _patch(source, output, sync_dir)
    assert completed.returncode == 0, completed.stderr

    with zipfile.ZipFile(output) as archive:
        assert archive.read("my_submission/requirements.txt").decode() == FIXED_REQUIREMENTS
        assert archive.read("my_submission/runtime/cuda_preload.py").decode() == NEW_MODULE
    report = json.loads(completed.stdout)
    assert report["prefix"] == "my_submission/"


def test_the_source_archive_is_left_untouched(tmp_path: Path, sync_dir: Path) -> None:
    """差し替えに失敗しても提出可能な現物が残っていることが、この経路の前提。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)
    before = source.read_bytes()

    _patch(source, output, sync_dir)

    assert source.read_bytes() == before


def test_writing_over_the_source_is_refused(tmp_path: Path, sync_dir: Path) -> None:
    source = tmp_path / "submission.zip"
    _build_archive(source)

    completed = _patch(source, source, sync_dir)
    assert completed.returncode != 0
    assert "must differ" in completed.stderr


def test_an_archive_without_requirements_is_rejected(tmp_path: Path, sync_dir: Path) -> None:
    source = tmp_path / "submission.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("policy_server.py", ORIGINAL_SERVER)

    completed = _patch(source, tmp_path / "out.zip", sync_dir)
    assert completed.returncode != 0
    assert "requirements.txt" in completed.stderr


def test_a_deeply_nested_requirements_is_not_mistaken_for_the_real_one(
    tmp_path: Path, sync_dir: Path
) -> None:
    """`prismatic/.../requirements.txt`のような同名ファイルは提出物が読むものではない。"""
    source = tmp_path / "submission.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("policy_server.py", ORIGINAL_SERVER)
        archive.writestr("vendor/prismatic/extras/requirements.txt", "irrelevant\n")

    completed = _patch(source, tmp_path / "out.zip", sync_dir)
    assert completed.returncode != 0


def _load_module():
    spec = importlib.util.spec_from_file_location("patch_submission_zip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(source: Path, output: Path, sync_dir: Path) -> list[str]:
    return [
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(output),
        "--sync-dir",
        str(sync_dir),
    ]


def test_entries_past_the_zip64_threshold_are_written(
    tmp_path: Path, sync_dir: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(sys, "argv", _argv(source, output, sync_dir))
    module.main()

    with zipfile.ZipFile(output) as archive:
        assert archive.read("model_weights/model.safetensors") == WEIGHT_BYTES
        assert archive.read("requirements.txt").decode() == FIXED_REQUIREMENTS


def test_a_failed_patch_leaves_no_partial_archive(
    tmp_path: Path, sync_dir: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(sys, "argv", _argv(source, output, sync_dir))
    with pytest.raises(RuntimeError):
        module.main()

    assert not output.exists()


def test_the_report_records_both_hashes(tmp_path: Path, sync_dir: Path) -> None:
    """提出記録は差し替え前後の両方を残す。どのArchiveから作ったかが後で要る。"""
    source = tmp_path / "submission.zip"
    output = tmp_path / "submission_patched.zip"
    _build_archive(source)

    completed = _patch(source, output, sync_dir)
    report = json.loads(completed.stdout)

    assert report["input_sha256"] != report["sha256"]
    assert len(report["sha256"]) == 64
    assert report["entries"] == report["source_entries"] + len(report["added"])
    assert report["zip_bytes"] == output.stat().st_size
