#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
CONFIG_NAME="${CONFIG_NAME:-scrfd_1g}"
WORK_NAME="${WORK_NAME:-$CONFIG_NAME}"
GPU_ID="${GPU_ID:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SCRFD_TOTAL_EPOCHS="${SCRFD_TOTAL_EPOCHS:-}"
SCRFD_EVAL_INTERVAL="${SCRFD_EVAL_INTERVAL:-}"
SCRFD_CHECKPOINT_INTERVAL="${SCRFD_CHECKPOINT_INTERVAL:-}"
SCRFD_SAMPLES_PER_GPU="${SCRFD_SAMPLES_PER_GPU:-}"
SCRFD_WORKERS_PER_GPU="${SCRFD_WORKERS_PER_GPU:-}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

TRAIN_CFG_OPTIONS=("${SCRFD_CFG_OPTIONS[@]}")
if [[ -n "$SCRFD_TOTAL_EPOCHS" ]]; then
  TRAIN_CFG_OPTIONS+=("total_epochs=${SCRFD_TOTAL_EPOCHS}")
fi
if [[ -n "$SCRFD_EVAL_INTERVAL" ]]; then
  TRAIN_CFG_OPTIONS+=("evaluation.interval=${SCRFD_EVAL_INTERVAL}")
fi
if [[ -n "$SCRFD_CHECKPOINT_INTERVAL" ]]; then
  TRAIN_CFG_OPTIONS+=("checkpoint_config.interval=${SCRFD_CHECKPOINT_INTERVAL}")
fi
if [[ -n "$SCRFD_SAMPLES_PER_GPU" ]]; then
  TRAIN_CFG_OPTIONS+=("data.samples_per_gpu=${SCRFD_SAMPLES_PER_GPU}")
fi
if [[ -n "$SCRFD_WORKERS_PER_GPU" ]]; then
  TRAIN_CFG_OPTIONS+=("data.workers_per_gpu=${SCRFD_WORKERS_PER_GPU}")
fi

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
cmd+=(--cfg-options "${TRAIN_CFG_OPTIONS[@]}")

CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"
