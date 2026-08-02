#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_REPO="https://github.com/matsuolab/PARC2026_pre.git"
OFFICIAL_COMMIT="fcac626949f52c0edb4b0701bc75e87464e36d64"
TARGET_DIR="${1:-vendor/PARC2026_pre}"

if [ -d "$TARGET_DIR/.git" ]; then
  git -C "$TARGET_DIR" fetch --all --tags --prune
else
  mkdir -p "$(dirname "$TARGET_DIR")"
  git clone "$OFFICIAL_REPO" "$TARGET_DIR"
fi

git -C "$TARGET_DIR" checkout --detach "$OFFICIAL_COMMIT"

echo "Official PARC2026 repository prepared."
echo "Path: $TARGET_DIR"
echo "Commit: $(git -C "$TARGET_DIR" rev-parse HEAD)"
