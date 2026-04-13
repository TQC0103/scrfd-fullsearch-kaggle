#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
CONFIG_NAME="${CONFIG_NAME:-scrfd_2.5g_online_sr}"
WORK_NAME="${WORK_NAME:-scrfd_2.5g_online_sr_2gpu}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29501}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SCRFD_TOTAL_EPOCHS="${SCRFD_TOTAL_EPOCHS:-}"
SCRFD_EVAL_INTERVAL="${SCRFD_EVAL_INTERVAL:-}"
SCRFD_CHECKPOINT_INTERVAL="${SCRFD_CHECKPOINT_INTERVAL:-}"
SCRFD_SAMPLES_PER_GPU="${SCRFD_SAMPLES_PER_GPU:-8}"
SCRFD_WORKERS_PER_GPU="${SCRFD_WORKERS_PER_GPU:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ensure_scrfd_dirs
assert_scrfd_dataset
build_scrfd_cfg_options

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPUS="${#GPU_ARRAY[@]}"

TRAIN_CFG_OPTIONS=("${SCRFD_CFG_OPTIONS[@]}")
TRAIN_CFG_OPTIONS+=("data.samples_per_gpu=${SCRFD_SAMPLES_PER_GPU}")
TRAIN_CFG_OPTIONS+=("data.workers_per_gpu=${SCRFD_WORKERS_PER_GPU}")
TRAIN_CFG_OPTIONS+=("cudnn_benchmark=True")

if [[ -n "$SCRFD_TOTAL_EPOCHS" ]]; then
  TRAIN_CFG_OPTIONS+=("total_epochs=${SCRFD_TOTAL_EPOCHS}")
fi
if [[ -n "$SCRFD_EVAL_INTERVAL" ]]; then
  TRAIN_CFG_OPTIONS+=("evaluation.interval=${SCRFD_EVAL_INTERVAL}")
fi
if [[ -n "$SCRFD_CHECKPOINT_INTERVAL" ]]; then
  TRAIN_CFG_OPTIONS+=("checkpoint_config.interval=${SCRFD_CHECKPOINT_INTERVAL}")
fi

cd "$SCRFD_REPO_ROOT"

cmd=(
  bash tools/dist_train.sh
  "$(scrfd_config_path "$CONFIG_GROUP" "$CONFIG_NAME")"
  "$GPUS"
  --work-dir "$(scrfd_work_dir "$WORK_NAME")"
)

if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $EXTRA_TRAIN_ARGS )
  cmd+=("${extra_args[@]}")
fi

cmd+=("$@")
cmd+=(--cfg-options "${TRAIN_CFG_OPTIONS[@]}")

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
OMP_NUM_THREADS="$OMP_NUM_THREADS" \
MKL_NUM_THREADS="$MKL_NUM_THREADS" \
NCCL_DEBUG="$NCCL_DEBUG" \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
"${cmd[@]}"
