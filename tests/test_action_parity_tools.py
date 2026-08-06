from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURE = REPO_ROOT / "scripts/capture_action_chunks.py"
COMPARE = REPO_ROOT / "submission/openvla_oft_offline/tools/compare_action_chunks.py"


def _capture_module():
    spec = importlib.util.spec_from_file_location("capture", CAPTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observations_are_reproducible_across_runs() -> None:
    """Reference and candidate run in separate processes, so the same seed has to
    produce the same observations or the comparison is meaningless."""
    module = _capture_module()

    first = module.build_observations(4, seed=20260804, camera=128)
    second = module.build_observations(4, seed=20260804, camera=128)

    assert len(first) == len(second) == 4
    for a, b in zip(first, second):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    different = module.build_observations(4, seed=1, camera=128)
    assert not np.array_equal(first[0]["agentview_image"], different[0]["agentview_image"])


def test_observations_match_the_official_shapes() -> None:
    module = _capture_module()
    observation = module.build_observations(1, seed=0, camera=128)[0]

    assert observation["agentview_image"].shape == (128, 128, 3)
    assert observation["agentview_image"].dtype == np.uint8
    assert observation["robot0_eye_in_hand_image"].shape == (128, 128, 3)
    for key, size in [
        ("robot0_joint_pos", 7),
        ("robot0_eef_pos", 3),
        ("robot0_eef_quat", 4),
        ("robot0_gripper_qpos", 2),
    ]:
        assert observation[key].shape == (size,), key
        assert observation[key].dtype == np.float32, key


def test_attention_patch_can_be_skipped_for_the_reference_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fork implements bidirectional attention natively. Patching it on top
    would compare the reimplementation against itself and always agree."""
    from submission.openvla_oft_offline.runtime import model_runtime

    calls: list[str] = []
    monkeypatch.setattr(
        model_runtime, "apply_bidirectional_attention_patch", lambda: calls.append("patched")
    )
    monkeypatch.setattr(model_runtime, "configure_offline_environment", lambda: None)
    monkeypatch.setattr(model_runtime, "inspect_checkpoint", lambda path: object())
    monkeypatch.setattr(model_runtime.OpenVLAOfflineRuntime, "_load_model", lambda self: None)

    model_runtime.OpenVLAOfflineRuntime("unused", apply_attention_patch=False)
    assert calls == []

    model_runtime.OpenVLAOfflineRuntime("unused")  # submission default
    assert calls == ["patched"]


def _run_compare(tmp_path: Path, a: np.ndarray, b: np.ndarray, atol: str) -> tuple[int, dict]:
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    completed = subprocess.run(
        [sys.executable, str(COMPARE), str(tmp_path / "a.npy"), str(tmp_path / "b.npy"),
         "--atol", atol],
        capture_output=True, text=True, check=False,
    )
    payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    return completed.returncode, payload


def test_comparison_accepts_bfloat16_scale_noise(tmp_path: Path) -> None:
    reference = np.linspace(-1, 1, 8 * 7, dtype=np.float32).reshape(1, 8, 7)
    candidate = reference + 1e-5

    returncode, report = _run_compare(tmp_path, reference, candidate, "1e-3")
    assert returncode == 0
    assert report["passed"] is True


def test_comparison_rejects_an_implementation_divergence(tmp_path: Path) -> None:
    reference = np.linspace(-1, 1, 8 * 7, dtype=np.float32).reshape(1, 8, 7)
    candidate = reference.copy()
    candidate[0, 3, 2] += 0.5

    returncode, report = _run_compare(tmp_path, reference, candidate, "1e-3")
    assert returncode == 1
    assert report["passed"] is False
    assert report["max_abs_diff"] == pytest.approx(0.5, abs=1e-6)
