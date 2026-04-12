#!/usr/bin/env bash
set -euo pipefail

SCRFD_REPO_ROOT="${SCRFD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRFD_CONFIG_ROOT="${SCRFD_CONFIG_ROOT:-$SCRFD_REPO_ROOT/configs}"
SCRFD_DATA_ROOT="${SCRFD_DATA_ROOT:-$SCRFD_REPO_ROOT/data/retinaface}"
SCRFD_WORK_ROOT="${SCRFD_WORK_ROOT:-$SCRFD_REPO_ROOT/work_dirs}"
SCRFD_RESULT_ROOT="${SCRFD_RESULT_ROOT:-$SCRFD_REPO_ROOT/wouts}"
SCRFD_PYTHON="${SCRFD_PYTHON:-python}"
SCRFD_CHECKPOINT_NAME="${SCRFD_CHECKPOINT_NAME:-latest.pth}"

ensure_scrfd_dirs() {
  mkdir -p "$SCRFD_WORK_ROOT" "$SCRFD_RESULT_ROOT"
}

assert_scrfd_dataset() {
  local required=(
    "$SCRFD_DATA_ROOT/train/labelv2.txt"
    "$SCRFD_DATA_ROOT/train/images"
    "$SCRFD_DATA_ROOT/val/labelv2.txt"
    "$SCRFD_DATA_ROOT/val/images"
    "$SCRFD_DATA_ROOT/val/gt"
  )

  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing dataset path: $path" >&2
      echo "Set SCRFD_DATA_ROOT to your WIDER FACE / retinaface-format dataset root." >&2
      exit 1
    fi
  done
}

build_scrfd_cfg_options() {
  SCRFD_CFG_OPTIONS=(
    "data.train.ann_file=${SCRFD_DATA_ROOT}/train/labelv2.txt"
    "data.train.img_prefix=${SCRFD_DATA_ROOT}/train/images/"
    "data.val.ann_file=${SCRFD_DATA_ROOT}/val/labelv2.txt"
    "data.val.img_prefix=${SCRFD_DATA_ROOT}/val/images/"
    "data.test.ann_file=${SCRFD_DATA_ROOT}/val/labelv2.txt"
    "data.test.img_prefix=${SCRFD_DATA_ROOT}/val/images/"
  )
}

scrfd_config_path() {
  local group="$1"
  local name="$2"
  printf '%s/%s/%s.py' "$SCRFD_CONFIG_ROOT" "$group" "$name"
}

scrfd_work_dir() {
  local name="$1"
  printf '%s/%s' "$SCRFD_WORK_ROOT" "$name"
}

scrfd_checkpoint_path() {
  local name="$1"
  printf '%s/%s/%s' "$SCRFD_WORK_ROOT" "$name" "$SCRFD_CHECKPOINT_NAME"
}
