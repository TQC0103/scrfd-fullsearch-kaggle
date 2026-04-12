# Linux Virtualenv Setup

This repository is easier to run in a **Linux virtual environment you control**
than inside a locked Kaggle competition image.

For SCRFD, the most stable path is:

- Linux
- Python `3.8`
- PyTorch `1.7.0` + CUDA `11.0`
- TorchVision `0.8.1`
- `mmcv-full==1.2.7`

Why this stack:

- Official MMDetection `2.7.0` docs list compatibility with `mmcv-full>=1.1.5, <1.3` for that codebase.
- Official MMCV wheel indexes still expose ready-made `mmcv-full==1.2.7` wheels for `torch1.7.0/cu110`.
- This is much closer to the original SCRFD release era than modern Python `3.12` / PyTorch `2.10` environments.

## Fast Setup

From the repo root:

```bash
export PYTHON_BIN=python3.8
export VENV_DIR=$PWD/.venv-scrfd
bash scripts/setup_linux_venv.sh
```

Then activate:

```bash
source .venv-scrfd/bin/activate
```

Verify:

```bash
python -c "import torch, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__)"
```

Expected shape:

- `torch` around `1.7.0+cu110`
- `mmcv` around `1.2.7`
- `mmdet` imports successfully

## Baseline Path

Set dataset paths:

```bash
export SCRFD_DATA_ROOT=/path/to/retinaface
export SCRFD_WORK_ROOT=$PWD/work_dirs
export SCRFD_RESULT_ROOT=$PWD/wouts
```

Run the quick baseline:

```bash
bash scripts/train_scrfd_1g_quick.sh
```

Eval:

```bash
CHECKPOINT=$PWD/work_dirs/scrfd_1g_quick/latest.pth bash scripts/eval_scrfd_1g.sh
```

## Full Search Path

Once baseline is healthy:

```bash
bash scripts/search_step1_generate.sh
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_train.sh
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_eval.sh
IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_stat.sh
```

Then step 2:

```bash
TEMPLATE=10 bash scripts/search_step2_generate.sh
GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_train.sh
GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_eval.sh
IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_stat.sh
```

## Notes

- If your machine already has a newer NVIDIA driver, that is usually fine; the important part is matching the **PyTorch/MMCV build pair**, not downgrading the system driver.
- If `python3.8` is not installed, install it first via apt/pyenv/conda, then rerun the setup script.
- If you want a different CUDA/PyTorch pair, edit `TORCH_CHANNEL`, `TORCH_VERSION`, `TORCHVISION_VERSION`, and `MMCV_FULL_VERSION`, but keep it inside the official compatibility window.
