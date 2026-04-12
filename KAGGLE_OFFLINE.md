# Kaggle Competition Offline Workflow

This guide is for Kaggle competition notebooks where internet access is disabled, so `git clone` and normal `pip install` from PyPI/GitHub do not work.

## 1. Recommended setup

Use **two Kaggle Datasets**:

1. `scrfd-fullsearch-kaggle-src`
   - contains this repository as a plain folder or a zip archive
2. `scrfd-wheelhouse`
   - contains any wheel files that are missing from the competition image

This split makes it easier to update code without rebuilding the dependency bundle every time.

There is also a ready-made notebook template in `notebooks/kaggle_competition_click_run.ipynb`. Upload that notebook to Kaggle, attach the source dataset, and change only the small config block at the top.

## 2. Prepare the source bundle locally

From your local machine, run one of these before uploading to Kaggle:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_offline_bundle.ps1
```

or

```bash
bash scripts/package_offline_bundle.sh
```

That creates `dist/scrfd-fullsearch-kaggle-offline.zip`.

Upload either:

- the extracted repo folder as a Kaggle Dataset, or
- the generated zip file as a Kaggle Dataset asset

The bundle includes the ready-made notebook template under `notebooks/`.

## 3. In the competition notebook

If your Kaggle Dataset contains the zip:

```bash
!mkdir -p /kaggle/working/scrfd-fullsearch-kaggle
!unzip -q /kaggle/input/scrfd-fullsearch-kaggle-src/scrfd-fullsearch-kaggle-offline.zip -d /kaggle/working
%cd /kaggle/working/scrfd-fullsearch-kaggle
```

If your Kaggle Dataset contains the unpacked folder:

```bash
!cp -r /kaggle/input/scrfd-fullsearch-kaggle-src/scrfd-fullsearch-kaggle /kaggle/working/
%cd /kaggle/working/scrfd-fullsearch-kaggle
```

`/kaggle/input` is read-only, so always copy or unzip into `/kaggle/working` before training.

## 4. Use the repo without `pip install -e .`

If internet is off, the easiest way to use the local code is to run it in-place:

```bash
import os, sys
repo_root = "/kaggle/working/scrfd-fullsearch-kaggle"
sys.path.insert(0, repo_root)
os.environ["PYTHONPATH"] = repo_root + ":" + os.environ.get("PYTHONPATH", "")
```

You can also do it in bash:

```bash
export PYTHONPATH=/kaggle/working/scrfd-fullsearch-kaggle:$PYTHONPATH
```

This avoids needing to install the repo itself from the network.

## 5. Handle dependencies offline

Only the external packages need offline installation. There are two cases:

### Case A: competition image already has what you need

Check first:

```bash
import sys, torch
print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
```

If `mmcv`, `cv2`, and other dependencies are already available, you may not need any extra package installation.

### Case B: some packages are missing

Build a wheel bundle locally and upload it as a Kaggle Dataset.

Typical offline install pattern:

```bash
!pip install --no-index --find-links /kaggle/input/scrfd-wheelhouse -r requirements.txt
```

If you need a specific `mmcv-full` wheel, put that wheel in the same dataset and install it with:

```bash
!pip install --no-index --find-links /kaggle/input/scrfd-wheelhouse mmcv-full
```

Important: `mmcv-full` must match the competition image's Python, PyTorch, and CUDA versions.

## 6. Dataset paths

After the repo is copied into `/kaggle/working`, set:

```bash
%env SCRFD_DATA_ROOT=/kaggle/input/widerface-retinaface-format/retinaface
%env SCRFD_WORK_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/work_dirs
%env SCRFD_RESULT_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/wouts
```

## 7. Main offline run commands

Baseline sanity check:

```bash
!bash scripts/train_scrfd_1g_quick.sh
```

If you really want the longer default schedule from the bundled config, use:

```bash
!bash scripts/train_scrfd_1g.sh
!CHECKPOINT=/kaggle/working/scrfd-fullsearch-kaggle/work_dirs/scrfd_1g/latest.pth \
  bash scripts/eval_scrfd_1g.sh
```

Search step 1:

```bash
!bash scripts/search_step1_generate.sh
!GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_train.sh
!GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_eval.sh
!IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_stat.sh
```

Search step 2:

```bash
!TEMPLATE=10 bash scripts/search_step2_generate.sh
!GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_train.sh
!GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_eval.sh
!IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_stat.sh
```

## 8. Practical advice for competition notebooks

- Keep the source repo as a Kaggle Dataset input, not only in GitHub.
- Keep wheel files in a separate Kaggle Dataset if the image is missing dependencies.
- Save checkpoints to `/kaggle/working`, then export them to a persistent Kaggle Dataset if the run matters.
- If you iterate on search logic often, update only the source dataset instead of rebuilding everything.
