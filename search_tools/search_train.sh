#!/usr/bin/env bash
set -euo pipefail

GROUP="${GROUP:-scrfdgen2.5g}"
PREFIX="${PREFIX:-$GROUP}"
TASKS_PER_GPU="${TASKS_PER_GPU:-8}"
OFFSET="${OFFSET:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
LAUNCHER="${LAUNCHER:-single}"

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    start=$((TASKS_PER_GPU * gpu + OFFSET))
    end=$((TASKS_PER_GPU * (gpu + 1) + OFFSET - 1))
    echo "$gpu,$start,$end,$GROUP,$PREFIX"
    python -u search_tools/search_train.py "$gpu" "$start" "$end" "$GROUP" \
      --prefix "$PREFIX" \
      --launcher "$LAUNCHER" \
      "$@" > "gpu${gpu}.log" 2>&1 &
done

