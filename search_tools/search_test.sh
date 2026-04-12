#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-wouts}"
THR="${THR:-0.02}"
GROUP="${GROUP:-scrfdgen2.5g}"
PREFIX="${PREFIX:-$GROUP}"
IDX_FROM="${IDX_FROM:-1}"
IDX_TO="${IDX_TO:-320}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-work_dirs}"

for i in $(seq "$IDX_FROM" "$IDX_TO"); do
    task="${PREFIX}_${i}"
    checkpoint="${CHECKPOINT_ROOT}/${task}/latest.pth"
    echo "$task"
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/test_widerface.py \
      "./configs/${GROUP}/${task}.py" \
      "$checkpoint" \
      --mode 0 \
      --thr "$THR" \
      --out "${OUTPUT_DIR}/${GROUP}/${task}" \
      "$@"
done

