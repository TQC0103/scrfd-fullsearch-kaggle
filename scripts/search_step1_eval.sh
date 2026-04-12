#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GROUP="${GROUP:-scrfdgen2.5g}"
PREFIX="${PREFIX:-$GROUP}"
GPU_ID="${GPU_ID:-0}"
IDX_FROM="${IDX_FROM:-1}"
IDX_TO="${IDX_TO:-64}"
MODE="${MODE:-0}"
THR="${THR:-0.02}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

cd "$SCRFD_REPO_ROOT"

for idx in $(seq "$IDX_FROM" "$IDX_TO"); do
  task="${PREFIX}_${idx}"
  checkpoint="$(scrfd_checkpoint_path "$task")"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Skipping $task because checkpoint was not found: $checkpoint" >&2
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$SCRFD_PYTHON" tools/test_widerface.py \
    "$(scrfd_config_path "$GROUP" "$task")" \
    "$checkpoint" \
    --mode "$MODE" \
    --thr "$THR" \
    --out "$SCRFD_RESULT_ROOT/$GROUP/$task" \
    "$@" \
    --cfg-options "${SCRFD_CFG_OPTIONS[@]}"
done
