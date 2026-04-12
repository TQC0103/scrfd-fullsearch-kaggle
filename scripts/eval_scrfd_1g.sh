#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_GROUP=scrfd CONFIG_NAME=scrfd_1g WORK_NAME=scrfd_1g OUTPUT_NAME=scrfd_1g "$SCRIPT_DIR/eval_baseline.sh" "$@"
