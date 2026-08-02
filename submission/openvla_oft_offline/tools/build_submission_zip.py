from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

MAX_ZIP_BYTES = 20 * 1024**3
MAX_EXTRACTED_BYTES = 40 * 1024**3
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "lora_adapter",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed in submission: {path}")
        if path.is_file():
            yield path, relative


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    required = (source / "policy_server.py", source / "requirements.txt", source / "prismatic")
    for path in required:
        if not path.exists():
            raise SystemExit(f"Required submission item missing: {path}")

    files = list(iter_files(source))
    extracted_bytes = sum(path.stat().st_size for path, _ in files)
    if extracted_bytes > MAX_EXTRACTED_BYTES:
        raise SystemExit(f"Extracted size exceeds 40 GiB: {extracted_bytes / 1024**3:.2f} GiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve().is_relative_to(source):
        raise SystemExit("Output zip must be outside the submission source directory")

    with zipfile.ZipFile(args.output, "w", allowZip64=True) as archive:
        for path, relative in files:
            compression = (
                zipfile.ZIP_STORED
                if path.suffix in {".safetensors", ".pt", ".bin"}
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, relative.as_posix(), compress_type=compression)

    zip_bytes = args.output.stat().st_size
    if zip_bytes > MAX_ZIP_BYTES:
        args.output.unlink(missing_ok=True)
        raise SystemExit(f"ZIP exceeds 20 GiB: {zip_bytes / 1024**3:.2f} GiB")

    report = {
        "source": str(source),
        "output": str(args.output.resolve()),
        "entries": len(files),
        "zip_bytes": zip_bytes,
        "zip_gib": zip_bytes / 1024**3,
        "extracted_bytes": extracted_bytes,
        "extracted_gib": extracted_bytes / 1024**3,
        "sha256": sha256(args.output),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
