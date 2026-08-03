#!/usr/bin/env bash
set -euo pipefail

# Official LIBERO master resolved on 2026-08-03. Override explicitly when upgrading.
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
LIBERO_WORKDIR="${LIBERO_WORKDIR:-/content/LIBERO}"
OPENVLA_OFT_WORKDIR="${OPENVLA_OFT_WORKDIR:-/content/openvla-oft}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

rm -rf "$LIBERO_WORKDIR"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_WORKDIR"
git -C "$LIBERO_WORKDIR" checkout "$LIBERO_COMMIT"
python -m pip install -e "$LIBERO_WORKDIR"

LIBERO_REQUIREMENTS="$OPENVLA_OFT_WORKDIR/experiments/robot/libero/libero_requirements.txt"
if [[ -f "$LIBERO_REQUIREMENTS" ]]; then
  python -m pip install -r "$LIBERO_REQUIREMENTS"
fi

python - <<'PY'
import json
import os
import subprocess

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

result = {
    "libero_commit": subprocess.check_output(
        ["git", "-C", "/content/LIBERO", "rev-parse", "HEAD"], text=True
    ).strip(),
    "task_suites": sorted(benchmark.get_benchmark_dict()),
    "offscreen_env_imported": OffScreenRenderEnv is not None,
    "mujoco_gl": os.environ.get("MUJOCO_GL"),
    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
}
print(json.dumps(result, indent=2))
PY
