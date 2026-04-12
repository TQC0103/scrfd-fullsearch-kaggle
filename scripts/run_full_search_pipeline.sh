#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/search_step1_generate.sh"
bash "$SCRIPT_DIR/search_step1_train.sh"
bash "$SCRIPT_DIR/search_step1_eval.sh"
bash "$SCRIPT_DIR/search_step1_stat.sh"

bash "$SCRIPT_DIR/search_step2_generate.sh"
bash "$SCRIPT_DIR/search_step2_train.sh"
bash "$SCRIPT_DIR/search_step2_eval.sh"
bash "$SCRIPT_DIR/search_step2_stat.sh"
