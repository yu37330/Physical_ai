from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/assemble_submission_checkpoint.py"


@pytest.fixture
def base_checkpoint(tmp_path: Path) -> Path:
    """A base checkpoint shaped like the real one, including what must not ship."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type": "openvla"}', encoding="utf-8")
    (base / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "dataset_statistics.json").write_text('{"libero_spatial": {}}', encoding="utf-8")
    (base / "action_head--150000_checkpoint.pt").write_bytes(b"base head")
    (base / "proprio_projector--150000_checkpoint.pt").write_bytes(b"base proprio")
    (base / "model_source_manifest.json").write_text(
        json.dumps({"resolved_revision": "a" * 40}), encoding="utf-8"
    )
    # Excluded from the submission.
    (base / "optimizer.pt").write_bytes(b"training state")
    (base / "lora_adapter").mkdir()
    (base / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (base / ".cache").mkdir()
    (base / ".cache" / "junk").write_bytes(b"junk")
    return base


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "stage_a_s1" / "openvla+mix+b8"
    run.mkdir(parents=True)
    (run / "action_head--100_checkpoint.pt").write_bytes(b"trained head")
    (run / "proprio_projector--100_checkpoint.pt").write_bytes(b"trained proprio")
    (run / "dataset_statistics.json").write_text(
        '{"parc_libero_plus_selected": {}}', encoding="utf-8"
    )
    return tmp_path / "runs" / "stage_a_s1"


def _assemble(base: Path, run: Path, output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--base-checkpoint", str(base),
         "--trained-run-dir", str(run),
         "--output", str(output)],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_trained_components_replace_the_base_ones(
    base_checkpoint: Path, run_dir: Path, tmp_path: Path
) -> None:
    """inspect_checkpoint requires exactly one of each, so the base copies have to
    go, and a submission carrying the base head would not be independently
    trained at all."""
    output = tmp_path / "out"
    report = _assemble(base_checkpoint, run_dir, output)

    assert (output / "action_head--100_checkpoint.pt").read_bytes() == b"trained head"
    assert not (output / "action_head--150000_checkpoint.pt").exists()
    assert (output / "proprio_projector--100_checkpoint.pt").read_bytes() == b"trained proprio"
    assert not (output / "proprio_projector--150000_checkpoint.pt").exists()
    assert report["layout"]["action_head"] == "action_head--100_checkpoint.pt"


def test_dataset_statistics_come_from_the_training_run(
    base_checkpoint: Path, run_dir: Path, tmp_path: Path
) -> None:
    """The head emits actions normalised against the statistics it trained on;
    unnormalising with the base checkpoint's would shift every action."""
    output = tmp_path / "out"
    report = _assemble(base_checkpoint, run_dir, output)

    statistics = json.loads((output / "dataset_statistics.json").read_text(encoding="utf-8"))
    assert list(statistics) == ["parc_libero_plus_selected"]
    assert report["dataset_statistics_keys"] == ["parc_libero_plus_selected"]


def test_training_state_and_adapter_are_left_out(
    base_checkpoint: Path, run_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    _assemble(base_checkpoint, run_dir, output)

    assert not (output / "optimizer.pt").exists()
    assert not (output / "lora_adapter").exists()
    assert not (output / ".cache").exists()
    assert (output / "model-00001-of-00001.safetensors").exists()


def test_the_manifest_records_what_the_report_has_to_declare(
    base_checkpoint: Path, run_dir: Path, tmp_path: Path
) -> None:
    """OFFICIAL_RULES.md 9: base weight source and revision, checkpoint hashes,
    and what was trained."""
    output = tmp_path / "out"
    report = _assemble(base_checkpoint, run_dir, output)

    assert report["base_resolved_revision"] == "a" * 40
    assert len(report["trained_components"]["action_head"]["sha256"]) == 64
    assert len(report["trained_components"]["proprio_projector"]["sha256"]) == 64
    assert json.loads(
        (output / "parc_submission_manifest.json").read_text(encoding="utf-8")
    )["trained_run_dir"]


def test_the_latest_checkpoint_wins_when_several_were_kept(
    base_checkpoint: Path, run_dir: Path, tmp_path: Path
) -> None:
    nested = run_dir / "openvla+mix+b8"
    (nested / "action_head--500_checkpoint.pt").write_bytes(b"later head")
    (nested / "proprio_projector--500_checkpoint.pt").write_bytes(b"later proprio")

    output = tmp_path / "out"
    report = _assemble(base_checkpoint, run_dir, output)

    assert report["layout"]["action_head"] == "action_head--500_checkpoint.pt"
    assert (output / "action_head--500_checkpoint.pt").read_bytes() == b"later head"


def test_an_unnumbered_latest_checkpoint_is_usable(
    base_checkpoint: Path, tmp_path: Path
) -> None:
    """finetune.py writes an unnumbered '--latest' copy, and that is all that
    survives in the run directory persisted to Drive. Restoring from Drive after
    the runtime goes has to assemble from it."""
    run = tmp_path / "restored" / "openvla+mix+b8"
    run.mkdir(parents=True)
    (run / "action_head--latest_checkpoint.pt").write_bytes(b"trained head")
    (run / "proprio_projector--latest_checkpoint.pt").write_bytes(b"trained proprio")
    (run / "dataset_statistics.json").write_text(
        '{"parc_libero_plus_selected": {}}', encoding="utf-8"
    )

    output = tmp_path / "out"
    report = _assemble(base_checkpoint, tmp_path / "restored", output)

    assert report["layout"]["action_head"] == "action_head--latest_checkpoint.pt"
    assert (output / "action_head--latest_checkpoint.pt").read_bytes() == b"trained head"
    assert not (output / "action_head--150000_checkpoint.pt").exists()


def test_a_run_without_a_checkpoint_is_rejected(
    base_checkpoint: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty_run"
    empty.mkdir()

    completed = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--base-checkpoint", str(base_checkpoint),
         "--trained-run-dir", str(empty),
         "--output", str(tmp_path / "out")],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )

    assert completed.returncode != 0
    assert "action_head" in completed.stderr
