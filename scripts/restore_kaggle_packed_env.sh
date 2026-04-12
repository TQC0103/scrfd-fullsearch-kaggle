#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <packed-env.tar.gz> <target-dir>" >&2
  exit 1
fi

ENV_ARCHIVE="$1"
TARGET_DIR="$2"

mkdir -p "$TARGET_DIR"
tar -xzf "$ENV_ARCHIVE" -C "$TARGET_DIR"
"$TARGET_DIR/bin/python" "$TARGET_DIR/bin/conda-unpack"

echo "$TARGET_DIR/bin/python"
