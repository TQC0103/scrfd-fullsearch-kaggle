#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SEARCH_GROUP_PATH="${SEARCH_GROUP_PATH:-configs/scrfdgen2.5g}"
GFLOPS="${GFLOPS:-2.5}"
EPS="${EPS:-0.02}"
NUM_CONFIGS="${NUM_CONFIGS:-64}"
TEMPLATE="${TEMPLATE:-0}"

cd "$SCRFD_REPO_ROOT"

"$SCRFD_PYTHON" search_tools/generate_configs_2.5g.py \
  --group "$SEARCH_GROUP_PATH" \
  --template "$TEMPLATE" \
  --gflops "$GFLOPS" \
  --mode 1 \
  --eps "$EPS" \
  --num-configs "$NUM_CONFIGS" \
  "$@"
