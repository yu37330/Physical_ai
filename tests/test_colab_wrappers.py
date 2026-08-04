from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "training/openvla_oft_a100/scripts"
WRAPPERS = [
    "colab_preflight.sh",
    "colab_setup.sh",
    "colab_dataset_prepare.sh",
    "colab_stage_a.sh",
    "colab_submission_validate.sh",
    "colab_smoke.sh",
]

def _find_bash() -> str | None:
    """Return a bash that actually runs.

    On Windows `shutil.which("bash")` often resolves to the Windows Store
    app-execution alias, which exits with ERROR_PATH_NOT_FOUND instead of
    running anything, so every candidate is probed before being accepted.
    """
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"], capture_output=True, text=True, timeout=30, check=False
            )
        except OSError:
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return candidate
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no working bash available")


def _bash(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, *argv],
        cwd=REPO_ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        # The scripts print Japanese guidance; do not decode with the OS default.
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _run(script: str, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _bash([str(SCRIPTS / script), *args], env)


@pytest.fixture
def colab_env(tmp_path: Path) -> dict[str, str]:
    """A fake Colab layout: empty work root, mounted but empty Drive."""
    (tmp_path / "work").mkdir()
    (tmp_path / "drive").mkdir()
    # Forward slashes: bash treats the backslashes in a Windows path as glob
    # escapes, so the S1 checkpoint lookup would never match.
    return {
        "PROJECT_ROOT": REPO_ROOT.as_posix(),
        "WORK_ROOT": (tmp_path / "work").as_posix(),
        "DRIVE_ROOT": (tmp_path / "drive").as_posix(),
    }


@pytest.mark.parametrize("script", WRAPPERS)
def test_wrapper_scripts_parse(script: str) -> None:
    completed = _bash(["-n", str(SCRIPTS / script)])
    assert completed.returncode == 0, completed.stderr


def test_stage_a_rejects_unknown_stage(colab_env: dict[str, str]) -> None:
    completed = _run("colab_stage_a.sh", ["bogus"], colab_env)
    assert completed.returncode == 2
    assert "Usage:" in completed.stderr


def test_stage_a_refuses_without_converted_rlds(colab_env: dict[str, str]) -> None:
    completed = _run("colab_stage_a.sh", ["s1"], colab_env)
    assert completed.returncode == 1
    assert "RLDS builder directory not found" in completed.stderr


def test_stage_a_refuses_a_broken_openvla_environment(colab_env: dict[str, str]) -> None:
    """colab_action_parity.sh swaps the transformers fork for the PyPI build.
    Training on that would use causal attention while the submission runtime uses
    bidirectional, without erroring, so refuse before a long run starts."""
    builder = Path(colab_env["WORK_ROOT"]) / "rlds/mini/parc_libero_plus_selected/1.0.0"
    builder.mkdir(parents=True)
    (builder / "dataset_info.json").write_text("{}", encoding="utf-8")
    (Path(colab_env["WORK_ROOT"]) / "models/openvla_oft_plus_base").mkdir(parents=True)
    s1_checkpoint = Path(colab_env["WORK_ROOT"]) / "runs/stage_a_s1_head_proprio_100/step"
    s1_checkpoint.mkdir(parents=True)
    (s1_checkpoint / "action_head--100_checkpoint.pt").write_bytes(b"")

    completed = _run("colab_stage_a.sh", ["s2"], colab_env)

    assert completed.returncode == 1
    assert "not ready for training" in completed.stderr


def test_stage_a_refuses_s2_before_s1_checkpoint(colab_env: dict[str, str]) -> None:
    """運用原則: S1が失敗した場合はS2を実行しない."""
    builder = Path(colab_env["WORK_ROOT"]) / "rlds/mini/parc_libero_plus_selected/1.0.0"
    builder.mkdir(parents=True)
    (builder / "dataset_info.json").write_text("{}", encoding="utf-8")
    (Path(colab_env["WORK_ROOT"]) / "models/openvla_oft_plus_base").mkdir(parents=True)

    completed = _run("colab_stage_a.sh", ["s2"], colab_env)
    assert completed.returncode == 1
    assert "S2 must not run until S1 passes its gate" in completed.stderr


def test_dataset_prepare_rejects_unknown_profile(colab_env: dict[str, str]) -> None:
    selection = Path(colab_env["DRIVE_ROOT"]) / "40_experiments/datasets/libero_plus_selection_v001.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}", encoding="utf-8")

    completed = _run("colab_dataset_prepare.sh", [], {**colab_env, "DATASET_PROFILE": "weird"})
    assert completed.returncode == 2
    assert "DATASET_PROFILE must be" in completed.stderr


def test_dataset_prepare_requires_manifest_when_promoting(colab_env: dict[str, str]) -> None:
    selection = Path(colab_env["DRIVE_ROOT"]) / "40_experiments/datasets/libero_plus_selection_v001.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}", encoding="utf-8")

    completed = _run("colab_dataset_prepare.sh", [], {**colab_env, "DATASET_PROFILE": "full"})
    assert completed.returncode == 2
    assert "Set MANIFEST_FILE" in completed.stderr


def test_persist_refuses_large_artifacts(tmp_path: Path, colab_env: dict[str, str]) -> None:
    """Drive is nearly full, so a stray checkpoint must not be copied there."""
    large = Path(colab_env["WORK_ROOT"]) / "large.bin"
    large.write_bytes(b"\0" * (6 * 1024 * 1024))
    destination = Path(colab_env["DRIVE_ROOT"]) / "large.bin"

    script = (
        f'source "{SCRIPTS / "colab_env.sh"}"\n'
        f'colab::persist "{large}" "{destination}" 1\n'
    )
    completed = _bash(["-c", script], colab_env)
    assert completed.returncode == 1
    assert "Refusing to copy" in completed.stderr
    assert not destination.exists()


def test_notebooks_delegate_to_the_same_wrappers() -> None:
    """Notebook cells and Colab Terminal must not drift apart."""
    expected = {
        "00_environment_check.ipynb": "colab_preflight.sh",
        "01_model_feasibility.ipynb": "colab_setup.sh",
        "02_dataset_prepare.ipynb": "colab_dataset_prepare.sh",
        "03_stage_a_train.ipynb": "colab_stage_a.sh",
        "04_submission_validate.ipynb": "colab_submission_validate.sh",
    }
    for notebook, wrapper in expected.items():
        payload = json.loads((REPO_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
        sources = "".join(
            "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
        )
        assert wrapper in sources, f"{notebook} no longer calls {wrapper}"
        assert (SCRIPTS / wrapper).is_file()
