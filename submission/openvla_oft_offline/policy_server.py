"""PARC2026 policy server.

The server and serialization sections follow the organizer template. Only the
MyPolicy implementation connects the offline OpenVLA-OFT+ runtime.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        ...


class MyPolicy(BasePolicy):
    def __init__(self):
        from runtime.policy import OfflinePolicy

        model_dir = Path(__file__).resolve().parent / "model_weights" / "openvla_oft_plus"
        self._policy = OfflinePolicy(model_dir)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self._policy.get_action(obs)

    def reset(self, instruction: str = "") -> None:
        self._policy.reset(instruction)


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json

        data = json.loads(body)
        instruction = data.get("instruction", "")
    assert _policy is not None
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    assert _policy is not None
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
