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
    "colab_pipeline.sh",
    "colab_run_detached.sh",
    "colab_transfer_submission.sh",
    "colab_patch_submission.sh",
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


def test_shell_dataset_identifiers_match_the_python_contract() -> None:
    """colab_env.sh builds the Drive RLDS path from these; if they drift from
    rlds_contract.py the restore silently misses and an hour of conversion is
    repeated."""
    from src.data.rlds_contract import DATASET_NAME, DATASET_VERSION

    script = (SCRIPTS / "colab_env.sh").read_text(encoding="utf-8")
    assert f'DATASET_NAME="${{DATASET_NAME:-{DATASET_NAME}}}"' in script
    assert f'DATASET_VERSION="${{DATASET_VERSION:-{DATASET_VERSION}}}"' in script


def test_notebooks_delegate_to_the_same_wrappers() -> None:
    """Notebook cells and Colab Terminal must not drift apart."""
    expected = {
        "00_environment_check.ipynb": "colab_preflight.sh",
        "01_model_feasibility.ipynb": "colab_setup.sh",
        "02_dataset_prepare.ipynb": "colab_dataset_prepare.sh",
        "03_stage_a_train.ipynb": "colab_stage_a.sh",
        "04_submission_validate.ipynb": "colab_submission_validate.sh",
        "07_patch_submission_requirements.ipynb": "colab_patch_submission.sh",
    }
    for notebook, wrapper in expected.items():
        payload = json.loads((REPO_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
        sources = "".join(
            "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
        )
        assert wrapper in sources, f"{notebook} no longer calls {wrapper}"
        assert (SCRIPTS / wrapper).is_file()


def test_patch_submission_refuses_a_missing_archive(colab_env: dict[str, str]) -> None:
    """Naming the wrong archive must fail before anything copies 14GB."""
    completed = _run("colab_patch_submission.sh", [], colab_env)
    assert completed.returncode == 1
    assert "Submission archive not found" in completed.stderr


def test_patch_submission_refuses_requirements_without_the_pin(
    tmp_path: Path, colab_env: dict[str, str]
) -> None:
    """Patching an archive with a requirements.txt that still lacks the pin would
    produce an archive that fails exactly as the last submission did. The whole
    point of the run is that pin, so its absence is not something to discover
    after a 14GB upload."""
    archive = tmp_path / "drive/60_submissions/parc2026_track1_openvla_oft_plus.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"not really a zip")
    stale = tmp_path / "requirements.txt"
    stale.write_text("torch==2.2.0\n", encoding="utf-8")

    completed = _run(
        "colab_patch_submission.sh", [], {**colab_env, "REQUIREMENTS": stale.as_posix()}
    )
    assert completed.returncode == 1
    assert "nvidia-nvjitlink-cu12" in completed.stderr


def test_detached_runner_rejects_a_missing_wrapper(colab_env: dict[str, str]) -> None:
    """The point is to walk away from the run, so a typo has to fail now rather
    than detach into a log nobody is watching."""
    completed = _run("colab_run_detached.sh", ["DATASET_PROFILE=full", "nope.sh"], colab_env)

    assert completed.returncode == 1
    assert "Wrapper not found" in completed.stderr


def test_detached_runner_requires_a_wrapper_after_the_assignments(
    colab_env: dict[str, str],
) -> None:
    completed = _run("colab_run_detached.sh", ["DATASET_PROFILE=full"], colab_env)

    assert completed.returncode == 2
    assert "No wrapper given" in completed.stderr


def test_pipeline_rejects_an_unknown_stage(colab_env: dict[str, str]) -> None:
    """It runs unattended for close to two hours, so a typo has to fail at the
    top rather than after setup has already spent twenty minutes."""
    completed = _run("colab_pipeline.sh", [], {**colab_env, "STAGES": "setup bogus"})

    assert completed.returncode == 2
    assert "Unknown stage: bogus" in completed.stderr


def test_pipeline_defaults_to_the_full_dataset_profile(colab_env: dict[str, str]) -> None:
    """colab_env.sh binds DATASET_PROFILE to mini, so a default set after
    sourcing it looks right and does nothing -- the pipeline would train on
    three episodes without saying so."""
    completed = _run("colab_pipeline.sh", [], {**colab_env, "STAGES": "setup"})

    assert "Profile: full" in completed.stdout


def test_pipeline_honours_an_explicit_profile(colab_env: dict[str, str]) -> None:
    completed = _run(
        "colab_pipeline.sh", [], {**colab_env, "STAGES": "setup", "DATASET_PROFILE": "mini"}
    )

    assert "Profile: mini" in completed.stdout


def test_tree_bytes_ignores_directory_inodes(tmp_path: Path, colab_env: dict[str, str]) -> None:
    """du -sb counts the directories themselves, which ext4 reports as 4096 and
    the Drive FUSE mount as 0. Comparing a restored tree against its source that
    way makes a complete copy look short by 4096 per directory."""
    tree = tmp_path / "tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "one.bin").write_bytes(b"\0" * 1000)
    (tree / "a" / "b" / "two.bin").write_bytes(b"\0" * 24)

    script = (
        f'source "{SCRIPTS / "colab_env.sh"}"\n'
        f'colab::tree_bytes "{tree.as_posix()}"\n'
    )
    completed = _bash(["-c", script], colab_env)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1024"


@pytest.mark.skipif(os.name == "nt", reason="needs sparse files to size a tree past 2^31")
def test_tree_bytes_does_not_use_scientific_notation(
    tmp_path: Path, colab_env: dict[str, str]
) -> None:
    """Colab runs mawk, which formats with OFMT %.6g and renders a 13GB total as
    1.28926e+10. Shell arithmetic cannot read that back, and the restore
    condition then reads false, which means 'convert from scratch'."""
    tree = tmp_path / "tree"
    tree.mkdir()
    big = tree / "shard.tfrecord"
    with big.open("wb") as handle:
        handle.truncate(12_892_612_026)

    script = (
        f'source "{SCRIPTS / "colab_env.sh"}"\n'
        f'total=$(colab::tree_bytes "{tree.as_posix()}")\n'
        f'(( total > 0 )) && echo "$total"\n'
    )
    completed = _bash(["-c", script], colab_env)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "12892612026"


def _restore_run(name: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        f'source "{SCRIPTS / "colab_env.sh"}"\n'
        f'colab::restore_run "{name}"\n'
    )
    return _bash(["-c", script], env)


def test_restore_run_brings_a_persisted_run_back_from_drive(
    colab_env: dict[str, str],
) -> None:
    """/content is recycled with the VM but every run persists to Drive. Without
    this, a fresh runtime is told to redo training that has already been done."""
    drive_run = Path(colab_env["DRIVE_ROOT"]) / "40_experiments/stage_a_s1_head_proprio_100/step"
    drive_run.mkdir(parents=True)
    (drive_run / "action_head--latest_checkpoint.pt").write_bytes(b"trained")

    completed = _restore_run("stage_a_s1_head_proprio_100", colab_env)

    assert completed.returncode == 0, completed.stderr
    restored = (
        Path(colab_env["WORK_ROOT"])
        / "runs/stage_a_s1_head_proprio_100/step/action_head--latest_checkpoint.pt"
    )
    assert restored.read_bytes() == b"trained"


def test_restore_run_reports_when_neither_copy_has_a_checkpoint(
    colab_env: dict[str, str],
) -> None:
    """A run directory on Drive without a checkpoint means the stage failed, not
    that it can be restored; the caller has to hear the difference."""
    empty = Path(colab_env["DRIVE_ROOT"]) / "40_experiments/stage_a_s1_head_proprio_100"
    empty.mkdir(parents=True)

    completed = _restore_run("stage_a_s1_head_proprio_100", colab_env)

    assert completed.returncode == 1


def test_restore_run_leaves_a_local_run_alone(colab_env: dict[str, str]) -> None:
    local_run = Path(colab_env["WORK_ROOT"]) / "runs/stage_a_s1_head_proprio_100/step"
    local_run.mkdir(parents=True)
    (local_run / "action_head--latest_checkpoint.pt").write_bytes(b"local")
    drive_run = Path(colab_env["DRIVE_ROOT"]) / "40_experiments/stage_a_s1_head_proprio_100/step"
    drive_run.mkdir(parents=True)
    (drive_run / "action_head--latest_checkpoint.pt").write_bytes(b"stale")

    completed = _restore_run("stage_a_s1_head_proprio_100", colab_env)

    assert completed.returncode == 0, completed.stderr
    assert (local_run / "action_head--latest_checkpoint.pt").read_bytes() == b"local"
