#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_GROUP="${CONFIG_GROUP:-scrfd}"
export CONFIG_NAME="${CONFIG_NAME:-scrfd_1g}"
export WORK_NAME="${WORK_NAME:-scrfd_1g_quick}"
export SCRFD_TOTAL_EPOCHS="${SCRFD_TOTAL_EPOCHS:-8}"
export SCRFD_EVAL_INTERVAL="${SCRFD_EVAL_INTERVAL:-2}"
export SCRFD_CHECKPOINT_INTERVAL="${SCRFD_CHECKPOINT_INTERVAL:-2}"

"$SCRIPT_DIR/train_baseline.sh" "$@"
