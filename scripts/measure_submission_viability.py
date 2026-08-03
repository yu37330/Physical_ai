#!/usr/bin/env python3
"""提出成立性を実測する。学習前にGo/No-Goを判定するためのGate。

運営の`validate_submission.py`は静的検査と起動スモークを行うが、Peak VRAMと
Track総時間予算を測らない。この2つは予選ルール上のHard Gateなので、ここで補う。

    python scripts/measure_submission_viability.py \\
      --submission-dir submission/openvla_oft_offline \\
      --output /content/work/submission_viability.json

観測のシリアライズは運営の`pipeline/remote_policy.py`と同一形式にしてある。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import requests

# 予選ルール（docs/OFFICIAL_RULES.md）のHard Gate。
ACT_TIMEOUT_SEC = 10.0
SERVER_TIMEOUT_SEC = 120.0
TRACK_TIMEOUT_SEC = 3600.0
L4_VRAM_BYTES = 24 * 1024**3
CAMERA = 128


def build_observation(rng: np.random.Generator, camera: int) -> dict[str, np.ndarray]:
    """運営validatorと同じキー・shape・dtype。値だけは退化を避けて乱数にする。"""
    return {
        "agentview_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
        "robot0_joint_pos": rng.uniform(-2.0, 2.0, 7).astype(np.float32),
        "robot0_eef_pos": rng.uniform(-0.5, 0.5, 3).astype(np.float32),
        "robot0_eef_quat": rng.uniform(-1.0, 1.0, 4).astype(np.float32),
        "robot0_gripper_qpos": rng.uniform(-0.05, 0.05, 2).astype(np.float32),
    }


def serialize_obs(obs: dict[str, np.ndarray]) -> bytes:
    return msgpack.packb(
        {
            key: {"data": arr.tobytes(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
            for key, arr in obs.items()
        },
        use_bin_type=True,
    )


def deserialize_action(data: bytes) -> np.ndarray:
    return np.frombuffer(msgpack.unpackb(data, raw=False)["data"], dtype=np.float32).copy()


class VramSampler(threading.Thread):
    """nvidia-smiでGPU使用量をpollし、起動前からの増分の最大値を記録する。

    採点は単一GPUなので、プロセス単位ではなくGPU全体の増分で足りる。
    """

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.available = shutil.which("nvidia-smi") is not None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.total_bytes: int | None = None
        self._stop = threading.Event()

    def _query(self) -> tuple[int, int] | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        used, total = (int(part.strip()) for part in out.stdout.strip().splitlines()[0].split(","))
        return used * 1024**2, total * 1024**2

    def capture_baseline(self) -> None:
        if not self.available:
            return
        sample = self._query()
        if sample:
            self.baseline_bytes, self.total_bytes = sample

    def run(self) -> None:
        if not self.available:
            return
        while not self._stop.wait(self.interval):
            sample = self._query()
            if not sample:
                continue
            used, total = sample
            self.total_bytes = total
            self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes, used)

    def stop(self) -> None:
        self._stop.set()

    def report(self) -> dict[str, Any]:
        if not self.available:
            return {"measured": False, "reason": "nvidia-smi not found"}
        delta = None
        if self.peak_bytes is not None and self.baseline_bytes is not None:
            delta = max(0, self.peak_bytes - self.baseline_bytes)
        return {
            "measured": self.peak_bytes is not None,
            "baseline_bytes": self.baseline_bytes,
            "peak_used_bytes": self.peak_bytes,
            "peak_delta_bytes": delta,
            "gpu_total_bytes": self.total_bytes,
            "l4_budget_bytes": L4_VRAM_BYTES,
            "fits_l4_24gb": None if self.peak_bytes is None else self.peak_bytes < L4_VRAM_BYTES,
        }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, help="提出ZIP。指定すると容量Gateも判定する。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--camera", type=int, default=CAMERA)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=SERVER_TIMEOUT_SEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    submission_dir = args.submission_dir.resolve()
    server_script = submission_dir / "policy_server.py"
    if not server_script.is_file():
        raise SystemExit(f"policy_server.py not found: {server_script}")

    base_url = f"http://{args.host}:{args.port}"
    rng = np.random.default_rng(20260804)

    vram = VramSampler()
    vram.capture_baseline()

    print(f"Launching policy server from {submission_dir}", flush=True)
    process = subprocess.Popen(
        [sys.executable, "policy_server.py", "--host", args.host, "--port", str(args.port)],
        cwd=submission_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    vram.start()

    server_log: list[str] = []
    log_thread = threading.Thread(
        target=lambda: server_log.extend(iter(process.stdout.readline, "")), daemon=True
    )
    log_thread.start()

    report: dict[str, Any] = {"submission_dir": str(submission_dir)}
    session = requests.Session()
    try:
        # 1. Cold start
        started = time.perf_counter()
        cold_start = None
        while time.perf_counter() - started < args.startup_timeout:
            if process.poll() is not None:
                raise SystemExit(
                    "Policy server exited before becoming healthy:\n" + "".join(server_log[-60:])
                )
            try:
                if session.get(f"{base_url}/health", timeout=5).status_code == 200:
                    cold_start = time.perf_counter() - started
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        if cold_start is None:
            raise SystemExit(
                f"/health did not return 200 within {args.startup_timeout}s:\n"
                + "".join(server_log[-60:])
            )
        print(f"Cold start: {cold_start:.2f}s", flush=True)

        # 2. /reset
        reset_started = time.perf_counter()
        session.post(
            f"{base_url}/reset",
            json={"instruction": "pick up the black bowl and place it on the plate"},
            timeout=ACT_TIMEOUT_SEC * 3,
        ).raise_for_status()
        reset_latency = time.perf_counter() - reset_started

        # 3. Warm-up は計測から除く（初回はCUDA context確保などで必ず遅い）
        for _ in range(args.warmup_requests):
            session.post(
                f"{base_url}/act",
                data=serialize_obs(build_observation(rng, args.camera)),
                headers={"Content-Type": "application/x-msgpack"},
                timeout=ACT_TIMEOUT_SEC * 6,
            ).raise_for_status()

        # 4. /act 本計測
        latencies: list[float] = []
        contract_failures: list[dict[str, Any]] = []
        for index in range(args.num_requests):
            payload = serialize_obs(build_observation(rng, args.camera))
            act_started = time.perf_counter()
            response = session.post(
                f"{base_url}/act",
                data=payload,
                headers={"Content-Type": "application/x-msgpack"},
                timeout=ACT_TIMEOUT_SEC * 6,
            )
            latencies.append(time.perf_counter() - act_started)
            response.raise_for_status()
            action = deserialize_action(response.content)
            # dtypeはサーバーの変更不可セクションが`astype(np.float32)`で強制するため
            # ここでは常に一致する。実際に落とせるのはshapeと非有限値。
            if action.shape != (7,) or action.dtype != np.float32 or not np.all(np.isfinite(action)):
                contract_failures.append(
                    {"index": index, "shape": list(action.shape),
                     "dtype": str(action.dtype), "finite": bool(np.all(np.isfinite(action)))}
                )
    finally:
        vram.stop()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    ordered = sorted(latencies)
    max_latency = ordered[-1]
    mean_latency = statistics.fmean(latencies)
    p95_latency = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    report["startup"] = {
        "cold_start_sec": cold_start,
        "limit_sec": SERVER_TIMEOUT_SEC,
        "pass": cold_start < SERVER_TIMEOUT_SEC,
    }
    report["latency"] = {
        "reset_sec": reset_latency,
        "act_max_sec": max_latency,
        "act_p95_sec": p95_latency,
        "act_mean_sec": mean_latency,
        "act_min_sec": ordered[0],
        "samples": len(latencies),
        "warmup_excluded": args.warmup_requests,
        "limit_sec": ACT_TIMEOUT_SEC,
        # 1回でも超えるとTrack全体が0点になるため、平均ではなく最大で判定する。
        "pass": max_latency < ACT_TIMEOUT_SEC and reset_latency < ACT_TIMEOUT_SEC,
    }
    # Track評価全体の1時間上限は10秒制限とは独立のGate。
    report["track_budget"] = {
        "limit_sec": TRACK_TIMEOUT_SEC,
        "act_per_second": 1.0 / mean_latency if mean_latency else None,
        "max_inferences_within_budget": int(TRACK_TIMEOUT_SEC / mean_latency) if mean_latency else None,
        "note": "環境ステップ・レンダリング時間は含まない推論のみの上限。実測の総Trial数と突き合わせること。",
    }
    report["action_contract"] = {
        "checked": len(latencies),
        "failures": contract_failures,
        "pass": not contract_failures,
    }
    report["vram"] = vram.report()

    report["size"] = {"submission_dir_bytes": directory_bytes(submission_dir)}
    if args.zip:
        zip_bytes = args.zip.stat().st_size
        report["size"].update(
            {
                "zip_bytes": zip_bytes,
                "zip_limit_bytes": 20 * 1024**3,
                "pass": zip_bytes < 20 * 1024**3,
            }
        )

    gates = [
        report["startup"]["pass"],
        report["latency"]["pass"],
        report["action_contract"]["pass"],
        report["vram"].get("fits_l4_24gb") is not False,
        report["size"].get("pass", True),
    ]
    report["status"] = "pass" if all(gates) else "fail"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
