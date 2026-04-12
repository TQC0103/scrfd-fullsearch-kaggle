#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GROUP="${GROUP:-scrfdgen2.5g}"
PREFIX="${PREFIX:-$GROUP}"
GPU_ID="${GPU_ID:-0}"
IDX_FROM="${IDX_FROM:-1}"
IDX_TO="${IDX_TO:-64}"
LAUNCHER="${LAUNCHER:-single}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

cd "$SCRFD_REPO_ROOT"

"$SCRFD_PYTHON" search_tools/search_train.py \
  "$GPU_ID" "$IDX_FROM" "$IDX_TO" "$GROUP" \
  --prefix "$PREFIX" \
  --launcher "$LAUNCHER" \
  --work-dir-root "$SCRFD_WORK_ROOT" \
  "$@" \
  --cfg-options "${SCRFD_CFG_OPTIONS[@]}"
