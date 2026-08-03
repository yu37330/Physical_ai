from __future__ import annotations

from pathlib import Path

from src.data.build_mini_selection import build_mini_selection
from training.openvla_oft_a100.scripts.build_checkpoint_manifest import build_manifest
from training.openvla_oft_a100.scripts.patch_openvla_oft_dependencies import patch as patch_dependencies

PINNED_PYPROJECT = """[project]
dependencies = [
    "draccus==0.8.0",
    "tensorflow==2.15.0",
    "tensorflow_datasets==4.9.3",
    "tensorflow_graphics==2021.12.3",
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
    }
    patched = pyproject.read_text(encoding="utf-8")
    assert '"tensorflow>=2.19,<2.20"' in patched
    assert '"tensorflow_datasets>=4.9.9,<4.10"' in patched
    assert "tensorflow_graphics" not in patched
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
