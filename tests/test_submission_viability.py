from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/measure_submission_viability.py"

pytest.importorskip("msgpack")
pytest.importorskip("requests")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

TEMPLATE_SERVER = '''
import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


class MyPolicy:
    def __init__(self):
        self.instruction = ""

    def get_action(self, obs):
{body}

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction


def deserialize_obs(data: bytes) -> dict:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {{}}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb({{"data": action.astype(np.float32).tobytes()}}, use_bin_type=True)


app = FastAPI()
_policy = MyPolicy()


@app.get("/health")
def health():
    return {{"status": "ok"}}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        instruction = json.loads(body).get("instruction", "")
    _policy.reset(instruction=instruction)
    return {{"status": "ok"}}


@app.post("/act")
async def act(request: Request):
    obs = deserialize_obs(await request.body())
    return Response(content=serialize_action(_policy.get_action(obs)),
                    media_type="application/x-msgpack")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
'''


def _write_server(directory: Path, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy_server.py").write_text(
        TEMPLATE_SERVER.format(body=body), encoding="utf-8"
    )


def _measure(directory: Path, port: int) -> tuple[int, dict]:
    output = directory / "report.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--submission-dir", str(directory),
         "--num-requests", "5", "--warmup-requests", "1",
         "--port", str(port), "--output", str(output)],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300, check=False,
    )
    if not output.is_file():
        pytest.fail(f"no report produced:\n{completed.stdout}\n{completed.stderr}")
    return completed.returncode, json.loads(output.read_text(encoding="utf-8"))


def test_conforming_policy_passes_every_gate(tmp_path: Path) -> None:
    _write_server(tmp_path, "        return np.zeros(7, dtype=np.float32)")
    returncode, report = _measure(tmp_path, 8231)

    assert returncode == 0, report
    assert report["status"] == "pass"
    assert report["startup"]["pass"] and report["latency"]["pass"]
    assert report["action_contract"]["pass"]
    assert report["latency"]["samples"] == 5
    assert report["track_budget"]["max_inferences_within_budget"] > 0


def test_non_finite_action_fails_the_contract_gate(tmp_path: Path) -> None:
    """NaN が混ざった Action は提出時に error 扱いになるため、ここで落とす。"""
    _write_server(
        tmp_path,
        "        a = np.zeros(7, dtype=np.float32)\n        a[0] = np.nan\n        return a",
    )
    returncode, report = _measure(tmp_path, 8232)

    assert returncode == 1
    assert report["status"] == "fail"
    assert report["action_contract"]["pass"] is False
    assert all(item["finite"] is False for item in report["action_contract"]["failures"])


def test_wrong_shape_fails_the_contract_gate(tmp_path: Path) -> None:
    _write_server(tmp_path, "        return np.zeros(6, dtype=np.float32)")
    returncode, report = _measure(tmp_path, 8233)

    assert returncode == 1
    assert report["action_contract"]["pass"] is False
    assert report["action_contract"]["failures"][0]["shape"] == [6]


def test_observation_matches_the_official_contract(tmp_path: Path) -> None:
    """運営 pipeline/remote_policy.py と同じ形式で観測を送っていることを確認する。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("viability", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    obs = module.build_observation(np.random.default_rng(0), 128)
    assert obs["agentview_image"].shape == (128, 128, 3)
    assert obs["agentview_image"].dtype == np.uint8
    assert obs["robot0_eye_in_hand_image"].shape == (128, 128, 3)
    for key, size in [
        ("robot0_joint_pos", 7),
        ("robot0_eef_pos", 3),
        ("robot0_eef_quat", 4),
        ("robot0_gripper_qpos", 2),
    ]:
        assert obs[key].shape == (size,), key
        assert obs[key].dtype == np.float32, key

    # Round-trip through the server's own deserializer shape.
    import msgpack

    unpacked = msgpack.unpackb(module.serialize_obs(obs), raw=False)
    for key, value in unpacked.items():
        assert set(value) == {"data", "shape", "dtype"}
        restored = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        assert restored.reshape(value["shape"]).shape == obs[key].shape
