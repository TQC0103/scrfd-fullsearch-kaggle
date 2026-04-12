#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BIN="${CONDA_BIN:-conda}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/environment.linux.yml}"
ENV_NAME="${ENV_NAME:-scrfd}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Missing conda. Install Miniconda/Mambaforge first, or override CONDA_BIN." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

echo "Using conda binary: ${CONDA_BIN}"
echo "Using environment file: ${ENV_FILE}"
echo "Environment name: ${ENV_NAME}"

"${CONDA_BIN}" env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
"${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install \
  -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html \
  mmcv-full==1.2.7
python -m pip install -e "${REPO_ROOT}"

cat <<EOF

Environment is ready.

Activate:
  conda activate ${ENV_NAME}

Sanity check:
  python -c "import torch, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__)"

Run baseline:
  export SCRFD_DATA_ROOT=/path/to/retinaface
  export SCRFD_WORK_ROOT=${REPO_ROOT}/work_dirs
  export SCRFD_RESULT_ROOT=${REPO_ROOT}/wouts
  bash ${REPO_ROOT}/scripts/train_scrfd_1g_quick.sh

EOF

