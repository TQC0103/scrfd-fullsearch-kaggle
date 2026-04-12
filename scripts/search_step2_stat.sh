#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GROUP="${GROUP:-scrfdgen2.5g}"
PREFIX="${PREFIX:-$GROUP}"
IDX_FROM="${IDX_FROM:-65}"
IDX_TO="${IDX_TO:-128}"

cd "$SCRFD_REPO_ROOT"

"$SCRFD_PYTHON" search_tools/search_stat.py \
  --group "$GROUP" \
  --prefix "$PREFIX" \
  --idx-from "$IDX_FROM" \
  --idx-to "$IDX_TO" \
  --result-dir "$SCRFD_RESULT_ROOT" \
  "$@"
