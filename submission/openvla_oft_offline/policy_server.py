from __future__ import annotations

import argparse
import os
from pathlib import Path

import json_numpy
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from runtime.offline_env import configure_offline_environment

configure_offline_environment()
json_numpy.patch()

from runtime.policy import OfflinePolicy


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("PARC_MODEL_DIR", ROOT / "model_weights" / "openvla_oft_plus"))

app = FastAPI()
policy: OfflinePolicy | None = None


@app.on_event("startup")
def startup() -> None:
    global policy
    policy = OfflinePolicy(MODEL_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if policy is None:
        return JSONResponse({"error": "policy not initialized"}, status_code=503)
    payload = await request.json()
    policy.reset(str(payload.get("instruction", "")), payload.get("seed"))
    return JSONResponse({"status": "ok"})


@app.post("/act")
async def act(request: Request) -> Response:
    if policy is None:
        return JSONResponse({"error": "policy not initialized"}, status_code=503)
    packed = await request.body()
    observation = msgpack.unpackb(packed, raw=False)
    action = np.asarray(policy.get_action(observation), dtype=np.float32).reshape(7)
    return Response(
        content=msgpack.packb(action, use_bin_type=True),
        media_type="application/msgpack",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
