from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest against the repository JSON Schema")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()

    try:
        import jsonschema
    except ImportError as exc:
        raise SystemExit("Install jsonschema to run manifest validation") from exc

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(manifest, schema)
    print(json.dumps({"valid": True, "dataset_id": manifest["dataset_id"]}, indent=2))


if __name__ == "__main__":
    main()
