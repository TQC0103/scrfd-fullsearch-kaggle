#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"
BUNDLE_NAME="${2:-scrfd-fullsearch-kaggle-offline.zip}"
STAGE_DIR="$OUTPUT_DIR/bundle_stage"
REPO_NAME="$(basename "$REPO_ROOT")"
ZIP_PATH="$OUTPUT_DIR/$BUNDLE_NAME"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/$REPO_NAME"
mkdir -p "$OUTPUT_DIR"

shopt -s dotglob
for path in "$REPO_ROOT"/*; do
  name="$(basename "$path")"
  case "$name" in
    .git|work_dirs|wouts|outputs|logs|tmp|dist|__pycache__)
      continue
      ;;
  esac
  cp -R "$path" "$STAGE_DIR/$REPO_NAME/"
done
shopt -u dotglob

find "$STAGE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

rm -f "$ZIP_PATH"
(
  cd "$STAGE_DIR"
  zip -qr "$ZIP_PATH" "$REPO_NAME"
)
rm -rf "$STAGE_DIR"

echo "Created offline bundle: $ZIP_PATH"
