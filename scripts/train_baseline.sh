#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
CONFIG_NAME="${CONFIG_NAME:-scrfd_1g}"
WORK_NAME="${WORK_NAME:-$CONFIG_NAME}"
GPU_ID="${GPU_ID:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

cd "$SCRFD_REPO_ROOT"

cmd=(
  "$SCRFD_PYTHON" tools/train.py
  "$(scrfd_config_path "$CONFIG_GROUP" "$CONFIG_NAME")"
  --gpu-ids 0
  --work-dir "$(scrfd_work_dir "$WORK_NAME")"
)

if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $EXTRA_TRAIN_ARGS )
  cmd+=("${extra_args[@]}")
fi

cmd+=("$@")
cmd+=(--cfg-options "${SCRFD_CFG_OPTIONS[@]}")

CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"
