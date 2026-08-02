#!/usr/bin/env bash
set -euo pipefail

OPENVLA_OFT_SOURCE="${OPENVLA_OFT_SOURCE:?Set OPENVLA_OFT_SOURCE to a pinned openvla-oft checkout}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rm -rf "$ROOT/prismatic"
cp -R "$OPENVLA_OFT_SOURCE/prismatic" "$ROOT/prismatic"
cp "$OPENVLA_OFT_SOURCE/LICENSE" "$ROOT/OPENVLA_OFT_LICENSE"

echo "Vendored prismatic runtime from: $OPENVLA_OFT_SOURCE"
echo "Destination: $ROOT/prismatic"
