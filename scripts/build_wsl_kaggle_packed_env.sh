#!/usr/bin/env bash
set -euo pipefail

# Build a Kaggle-friendly packed baseline environment from Linux/WSL.
# This targets Kaggle's current Python 3.12 + CUDA 12.8 style runtime.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
ENV_NAME="${ENV_NAME:-scrfd-kaggle-baseline}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/dist/upload_ready/scrfd-kaggle-baseline-py312-cu128.tar.gz}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda not found at: $CONDA_BIN" >&2
  echo "Set CONDA_BIN to your Miniforge/Conda executable inside WSL." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

"$CONDA_BIN" create -y -n "$ENV_NAME" python=3.12 pip conda-pack

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  "$REPO_ROOT/dist/upload_ready/scrfd-wheelhouse/mmcv-1.4.0-py2.py3-none-any.whl" \
  "$REPO_ROOT/dist/upload_ready/scrfd-wheelhouse/addict-2.4.0-py3-none-any.whl" \
  "$REPO_ROOT/dist/upload_ready/scrfd-wheelhouse/yapf-0.43.0-py3-none-any.whl" \
  "$REPO_ROOT/dist/upload_ready/scrfd-wheelhouse/platformdirs-4.9.6-py3-none-any.whl" \
  "$REPO_ROOT/dist/upload_ready/scrfd-wheelhouse/terminaltables-3.1.10-py2.py3-none-any.whl" \
  opencv-python-headless \
  matplotlib \
  six \
  tqdm \
  albumentations \
  pycocotools

"$CONDA_BIN" run -n "$ENV_NAME" conda-pack -o "$OUTPUT_PATH"

echo "Packed environment written to:"
echo "  $OUTPUT_PATH"
