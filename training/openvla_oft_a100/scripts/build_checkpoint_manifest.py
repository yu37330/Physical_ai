from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or Path(path).match(pattern) for pattern in patterns)


def _role(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return "model_weights"
    if "action_head" in lower and lower.endswith(".pt"):
        return "action_head"
    if "proprio_projector" in lower and lower.endswith(".pt"):
        return "proprio_projector"
    if "dataset_statistics" in lower:
        return "dataset_statistics"
    if "tokenizer" in lower or "special_tokens" in lower or "added_tokens" in lower:
        return "tokenizer"
    if lower.endswith("config.json") or "processor" in lower or "preprocessor" in lower:
        return "configuration"
    if "lora" in lower or "adapter" in lower:
        return "adapter"
    if any(token in lower for token in ("optimizer", "scheduler", "trainer_state", "rng_state")):
        return "training_state"
    return "support"


def build_manifest(
    *,
    model_root: Path,
    config: dict[str, Any],
    source_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    training_patterns = list(config.get("training_only_patterns", []))
    exclude_patterns = list(config.get("submission_exclude_patterns", []))
    files: list[dict[str, Any]] = []
    symlinks: list[str] = []

    for path in sorted(model_root.rglob("*")):
        if path.is_symlink():
            symlinks.append(path.relative_to(model_root).as_posix())
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(model_root).as_posix()
        role = _role(relative)
        training_only = _matches(relative, training_patterns) or role == "training_state"
        excluded = _matches(relative, exclude_patterns) or training_only
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "role": role,
                "training_only": training_only,
                "include_in_submission": not excluded,
            }
        )

    present_paths = [item["path"] for item in files]
    required_patterns = list(config.get("required_patterns", []))
    required_results = {
        pattern: [path for path in present_paths if fnmatch.fnmatch(path, pattern) or Path(path).match(pattern)]
        for pattern in required_patterns
    }
    missing_required = [pattern for pattern, matches in required_results.items() if not matches]

    total_bytes = sum(item["size_bytes"] for item in files)
    submission_bytes = sum(
        item["size_bytes"] for item in files if item["include_in_submission"]
    )
    training_state_files = [item["path"] for item in files if item["training_only"]]
    model_shards = [item["path"] for item in files if item["role"] == "model_weights"]

    source = dict(config.get("source", {}))
    if source_manifest:
        source.update(
            {
                key: source_manifest[key]
                for key in ("repo_id", "requested_revision", "resolved_revision")
                if key in source_manifest
            }
        )
    resolved_revision = source.get("resolved_revision")
    budgets = config.get("budgets", {})
    max_model_bytes = int(budgets.get("model_directory_max_bytes", 19_000_000_000))

    checks = {
        "required_files_present": not missing_required,
        "resolved_revision_present": bool(resolved_revision and resolved_revision != "main"),
        "no_symlinks": not symlinks,
        "no_training_state_in_submission": all(
            not item["include_in_submission"] for item in files if item["training_only"]
        ),
        "model_directory_within_budget": total_bytes <= max_model_bytes,
        "submission_payload_within_model_budget": submission_bytes <= max_model_bytes,
        "model_shards_present": bool(model_shards),
        "sha256_complete": all(len(item["sha256"]) == 64 for item in files),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "manifest_version": "1.0.0",
        "model_id": config["model_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model_root": str(model_root.resolve()),
        "files": files,
        "summary": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "submission_file_count": sum(item["include_in_submission"] for item in files),
            "submission_bytes": submission_bytes,
            "model_shard_count": len(model_shards),
            "training_state_file_count": len(training_state_files),
        },
        "required_pattern_matches": required_results,
        "missing_required_patterns": missing_required,
        "symlinks": symlinks,
        "budgets": budgets,
        "checks": checks,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible local checkpoint manifest")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_manifest = None
    if args.source_manifest and args.source_manifest.is_file():
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    payload = build_manifest(
        model_root=args.model_root,
        config=config,
        source_manifest=source_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    if payload["status"] != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
