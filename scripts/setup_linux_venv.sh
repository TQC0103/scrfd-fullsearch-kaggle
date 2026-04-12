#!/usr/bin/env bash
set -euo pipefail

if [[ "${OSTYPE:-}" != linux* ]]; then
  echo "This setup script is intended for Linux only." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.8}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-scrfd}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu110}"
TORCH_VERSION="${TORCH_VERSION:-1.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.8.1}"
MMCV_FULL_VERSION="${MMCV_FULL_VERSION:-1.2.7}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.8 first or override PYTHON_BIN." >&2
  exit 1
fi

echo "Repo root: ${REPO_ROOT}"
echo "Python bin: ${PYTHON_BIN}"
echo "Virtualenv: ${VENV_DIR}"
echo "Torch: ${TORCH_VERSION}+${TORCH_CHANNEL}"
echo "TorchVision: ${TORCHVISION_VERSION}+${TORCH_CHANNEL}"
echo "MMCV full: ${MMCV_FULL_VERSION}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy<2" cython

python -m pip install \
  "torch==${TORCH_VERSION}+${TORCH_CHANNEL}" \
  "torchvision==${TORCHVISION_VERSION}+${TORCH_CHANNEL}" \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install \
  "mmcv-full==${MMCV_FULL_VERSION}" \
  -f "https://download.openmmlab.com/mmcv/dist/${TORCH_CHANNEL}/torch${TORCH_VERSION}/index.html"

python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip install -e "${REPO_ROOT}"

cat <<EOF

SCRFD Linux virtualenv is ready.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Quick checks:
  python -c "import torch, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__)"
  bash "${REPO_ROOT}/scripts/train_scrfd_1g_quick.sh"

EOF
