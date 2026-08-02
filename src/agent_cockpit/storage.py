"""Google Drive互換のファイルTrace保存。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TraceStore:
    """Run/Step単位でJSON Traceと観測Artifactを永続化する。"""

    root: Path

    @classmethod
    def from_environment(cls) -> "TraceStore":
        drive_root = Path(
            os.environ.get(
                "PHYSICAL_AI_DRIVE_ROOT",
                "/content/drive/MyDrive/PARC2026",
            )
        )
        return cls(drive_root / "40_experiments" / "agent_cockpit")

    def run_dir(self, run_id: str) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def create_run(self, run_id: str, metadata: dict[str, Any]) -> Path:
        run_dir = self.run_dir(run_id)
        (run_dir / "steps").mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "official_evaluation_data_used_for_training": False,
            **metadata,
        }
        self._write_json(run_dir / "run_manifest.json", payload)
        return run_dir

    def step_dir(self, run_id: str, step_id: int) -> Path:
        step_dir = self.run_dir(run_id) / "steps" / f"step_{step_id:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def copy_artifact(
        self,
        run_id: str,
        step_id: int,
        source_path: str | Path,
        artifact_name: str,
    ) -> Path:
        """入力画像などをStep配下へコピーしてTraceと一緒に残す。"""

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact not found: {source}")
        destination = self.step_dir(run_id, step_id) / artifact_name
        shutil.copy2(source, destination)
        return destination

    def save_image_array(
        self,
        run_id: str,
        step_id: int,
        artifact_name: str,
        image: Any,
    ) -> Path:
        """uint8 RGB配列をPNGとしてStep配下へ保存する。"""

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        destination = self.step_dir(run_id, step_id) / artifact_name
        Image.fromarray(array, mode="RGB").save(destination)
        return destination

    def save_step_artifact(
        self,
        run_id: str,
        step_id: int,
        name: str,
        payload: dict[str, Any],
    ) -> Path:
        """既存Stepへ追加JSONを保存し、Timelineは重複追加しない。"""

        path = self.step_dir(run_id, step_id) / f"{name}.json"
        self._write_json(path, payload)
        return path

    def save_run_artifact(
        self,
        run_id: str,
        name: str,
        payload: dict[str, Any],
    ) -> Path:
        """Loop集計などRun全体のJSONを保存する。"""

        path = self.run_dir(run_id) / f"{name}.json"
        self._write_json(path, payload)
        return path

    def save_step(
        self,
        run_id: str,
        step_id: int,
        payloads: dict[str, dict[str, Any]],
    ) -> Path:
        step_dir = self.step_dir(run_id, step_id)
        for name, payload in payloads.items():
            self._write_json(step_dir / f"{name}.json", payload)
        self._append_jsonl(
            self.run_dir(run_id) / "timeline.jsonl",
            {
                "step_id": step_id,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": sorted(payloads),
            },
        )
        return step_dir

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
