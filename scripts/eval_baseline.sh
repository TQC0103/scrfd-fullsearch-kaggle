#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
CONFIG_NAME="${CONFIG_NAME:-scrfd_1g}"
WORK_NAME="${WORK_NAME:-$CONFIG_NAME}"
OUTPUT_NAME="${OUTPUT_NAME:-$WORK_NAME}"
GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-0}"
THR="${THR:-0.02}"
CHECKPOINT="${CHECKPOINT:-$(scrfd_checkpoint_path "$WORK_NAME")}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

cd "$SCRFD_REPO_ROOT"

cmd=(
  "$SCRFD_PYTHON" tools/test_widerface.py
  "$(scrfd_config_path "$CONFIG_GROUP" "$CONFIG_NAME")"
  "$CHECKPOINT"
  --mode "$MODE"
  --thr "$THR"
  --out "$SCRFD_RESULT_ROOT/$CONFIG_GROUP/$OUTPUT_NAME"
)

cmd+=("$@")
cmd+=(--cfg-options "${SCRFD_CFG_OPTIONS[@]}")

CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"
