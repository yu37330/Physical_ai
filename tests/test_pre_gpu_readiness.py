from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.data.build_mini_selection import build_mini_selection
from training.openvla_oft_a100.scripts.build_checkpoint_manifest import build_manifest
from training.openvla_oft_a100.scripts.patch_openvla_oft_dependencies import patch as patch_dependencies

ROOT = Path(__file__).resolve().parent.parent

PINNED_PYPROJECT = """[project]
dependencies = [
    "draccus==0.8.0",
    "tensorflow==2.15.0",
    "tensorflow_datasets==4.9.3",
    "tensorflow_graphics==2021.12.3",
    "dlimp @ git+https://github.com/moojink/dlimp_openvla",
    "sentencepiece==0.1.99",
    "timm==0.9.10",
]
"""


def test_build_mini_selection_uses_two_train_and_one_validation() -> None:
    episodes = []
    for episode_index in range(5):
        episodes.append(
            {
                "episode_index": episode_index,
                "instruction": "task a",
                "suite": "spatial",
                "split": "train" if episode_index < 3 else "validation",
            }
        )
    selection = {"selection_version": "1.0.0", "source": {}, "episodes": episodes}
    result = build_mini_selection(
        selection,
        instruction=None,
        train_count=2,
        validation_count=1,
    )
    assert result["counts"] == {"total": 3, "by_split": {"train": 2, "validation": 1}}
    assert [row["episode_index"] for row in result["episodes"]] == [0, 1, 3]


def test_dependency_patch_relaxes_tensorflow_pins_on_python_312(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PINNED_PYPROJECT, encoding="utf-8")

    applied = patch_dependencies(pyproject, (3, 12))
    assert applied == {
        "tensorflow==2.15.0": "tensorflow>=2.19,<2.20",
        "tensorflow_datasets==4.9.3": "tensorflow_datasets>=4.9.9,<4.10",
        # Dropped so pip stops resolving tensorflow-addons, which has no 3.12 wheel.
        "tensorflow_graphics==2021.12.3": None,
        # Dropped because dlimp pins tensorflow==2.15.0 itself; relaxing only the
        # top-level pin moves the conflict instead of resolving it.
        "dlimp @ git+https://github.com/moojink/dlimp_openvla": None,
        # 0.1.99 has no cp312 wheel, so 3.12 would fall back to a source build.
        "sentencepiece==0.1.99": "sentencepiece==0.2.0",
    }
    patched = pyproject.read_text(encoding="utf-8")
    assert '"tensorflow>=2.19,<2.20"' in patched
    assert '"tensorflow_datasets>=4.9.9,<4.10"' in patched
    assert "tensorflow_graphics" not in patched
    assert "dlimp" not in patched
    # Removing an entry must not leave a dangling line or comma behind.
    assert '    "timm==0.9.10",\n]' in patched
    # Unrelated pins keep the originally tested OpenVLA-OFT versions.
    assert '"draccus==0.8.0"' in patched

    # Re-running must not rewrite the already relaxed specifiers.
    assert patch_dependencies(pyproject, (3, 12)) == {}
    assert pyproject.read_text(encoding="utf-8") == patched


def test_dependency_patch_is_a_no_op_on_python_311(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PINNED_PYPROJECT, encoding="utf-8")

    assert patch_dependencies(pyproject, (3, 11)) == {}
    assert pyproject.read_text(encoding="utf-8") == PINNED_PYPROJECT


def _run_preflight(tmp_path: Path, extra: list[str]) -> dict:
    output = tmp_path / "preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "training/openvla_oft_a100/scripts/preflight.py"),
            "--project-root", str(ROOT),
            "--work-root", str(tmp_path),
            "--drive-root", str(tmp_path),
            "--output", str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,  # Exits 1 off a Colab GPU runtime; the payload is what matters.
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_preflight_makes_the_a100_check_opt_in(tmp_path: Path) -> None:
    """The T4 smoke reports the A100 check without being blocked by it."""
    without = _run_preflight(tmp_path / "a", ["--minimum-drive-free-gb", "0"])
    with_flag = _run_preflight(tmp_path / "b", ["--minimum-drive-free-gb", "0", "--require-a100-40gb"])

    assert "a100_40gb" in without["checks"]
    assert "a100_40gb" not in without["required_checks"]
    assert "a100_40gb" in with_flag["required_checks"]


def test_preflight_accepts_python_312(tmp_path: Path) -> None:
    """Current Colab runtimes are 3.12; the gate must not reject them."""
    payload = _run_preflight(tmp_path, ["--minimum-drive-free-gb", "0"])
    assert payload["checks"]["python_3_10_or_newer"] is True
    assert "python_3_10_or_3_11" not in payload["checks"]


def test_checkpoint_manifest_classifies_and_excludes_training_state(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    for relative, content in {
        "config.json": b"{}",
        "model-00001-of-00001.safetensors": b"weights",
        "tokenizer.json": b"{}",
        "special_tokens_map.json": b"{}",
        "action_head--1_checkpoint.pt": b"head",
        "proprio_projector--1_checkpoint.pt": b"proprio",
        "dataset_statistics.json": b"{}",
        "optimizer.pt": b"optimizer",
    }.items():
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    config = {
        "model_id": "test_model",
        "source": {"repo_id": "test/model", "requested_revision": "main"},
        "required_patterns": [
            "config.json",
            "model*.safetensors",
            "tokenizer*",
            "special_tokens_map.json",
            "action_head*checkpoint*.pt",
            "proprio_projector*checkpoint*.pt",
            "dataset_statistics.json",
        ],
        "training_only_patterns": ["optimizer*"],
        "submission_exclude_patterns": ["optimizer*"],
        "budgets": {"model_directory_max_bytes": 1_000_000},
    }
    result = build_manifest(
        model_root=model_root,
        config=config,
        source_manifest={
            "repo_id": "test/model",
            "requested_revision": "main",
            "resolved_revision": "abcdef1234567890",
        },
    )
    assert result["status"] == "pass"
    optimizer = next(item for item in result["files"] if item["path"] == "optimizer.pt")
    assert optimizer["training_only"] is True
    assert optimizer["include_in_submission"] is False
    assert result["summary"]["model_shard_count"] == 1
