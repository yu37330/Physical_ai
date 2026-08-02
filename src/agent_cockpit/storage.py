"""Google Drive互換のファイルTrace保存。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceStore:
    """Run/Step単位でJSON Traceを永続化する。"""

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

    def create_run(self, run_id: str, metadata: dict[str, Any]) -> Path:
        run_dir = self.root / run_id
        (run_dir / "steps").mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "official_evaluation_data_used_for_training": False,
            **metadata,
        }
        self._write_json(run_dir / "run_manifest.json", payload)
        return run_dir

    def save_step(
        self,
        run_id: str,
        step_id: int,
        payloads: dict[str, dict[str, Any]],
    ) -> Path:
        step_dir = self.root / run_id / "steps" / f"step_{step_id:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            self._write_json(step_dir / f"{name}.json", payload)
        self._append_jsonl(
            self.root / run_id / "timeline.jsonl",
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
