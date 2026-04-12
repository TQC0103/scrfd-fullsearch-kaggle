#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"
BUNDLE_NAME="${2:-scrfd-fullsearch-kaggle-offline.zip}"
python "$SCRIPT_DIR/package_offline_bundle.py" --repo-root "$REPO_ROOT" --output-dir "$OUTPUT_DIR" --bundle-name "$BUNDLE_NAME"
