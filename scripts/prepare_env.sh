#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --upgrade pip setuptools wheel
python -m pip install pycocotools
python -m pip install -r requirements/build.txt
python -m pip install matplotlib opencv-python-headless six terminaltables tqdm
python -m pip install -v -e .

cat <<'EOF'
Environment bootstrap completed.

If your Kaggle image does not already include mmcv/mmcv-full, install a wheel
that matches the active PyTorch + CUDA build before training or evaluation.
This repository intentionally keeps that step explicit because Kaggle images
change over time.

Search-only dependencies such as autotorch are intentionally excluded from this
bootstrap path because they are not required for baseline or online-SR training
and tend to break on newer Kaggle Python images.
EOF
