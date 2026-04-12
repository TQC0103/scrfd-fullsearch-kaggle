#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
export CONFIG_NAME="${CONFIG_NAME:-scrfd_1g}"
export WORK_NAME="${WORK_NAME:-scrfd_1g_quick}"
export SCRFD_TOTAL_EPOCHS="${SCRFD_TOTAL_EPOCHS:-16}"
export SCRFD_EVAL_INTERVAL="${SCRFD_EVAL_INTERVAL:-4}"
export SCRFD_CHECKPOINT_INTERVAL="${SCRFD_CHECKPOINT_INTERVAL:-4}"

bash "$SCRIPT_DIR/train_baseline.sh" "$@"
