# Kaggle Workflow Notes

This repository is designed so that a Kaggle notebook can clone it, point it at a WIDER FACE dataset, and continue with either a quick baseline sanity check or the full SCRFD two-step search workflow.

If your notebook comes from a Kaggle competition with internet disabled, switch to the offline workflow in `KAGGLE_OFFLINE.md` instead of using `git clone`.

## 1. Clone in a notebook

```bash
%cd /kaggle/working
!git clone https://github.com/TQC0103/scrfd-fullsearch-kaggle.git
%cd /kaggle/working/scrfd-fullsearch-kaggle
```

## 2. Install dependencies

Kaggle GPU images change over time, so install `mmcv-full` to match the PyTorch/CUDA build that Kaggle is using in your session.

```bash
!bash scripts/prepare_env.sh
```

If `mmcv-full` is not already available, install a wheel compatible with the active Kaggle image before training or evaluation. The repository keeps the rest of the dependencies in `requirements.txt`.

## 3. Point the repo at your dataset

The wrapper scripts use `SCRFD_DATA_ROOT`. A typical Kaggle notebook cell looks like:

```bash
%env SCRFD_DATA_ROOT=/kaggle/input/widerface-retinaface-format/retinaface
%env SCRFD_WORK_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/work_dirs
%env SCRFD_RESULT_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/wouts
```

Expected layout under `SCRFD_DATA_ROOT`:

```text
retinaface/
  train/
    images/
    labelv2.txt
  val/
    images/
    labelv2.txt
    gt/
      *.mat
```

## 4. Run a baseline sanity check

Train:

```bash
!bash scripts/train_scrfd_1g_quick.sh
```

Longer baseline schedule:

```bash
!bash scripts/train_scrfd_1g.sh
```

Evaluate:

```bash
!CHECKPOINT=/kaggle/working/scrfd-fullsearch-kaggle/work_dirs/scrfd_1g/latest.pth \
  bash scripts/eval_scrfd_1g.sh
```

## 5. Run the full search workflow

Step 1 candidate generation:

```bash
!bash scripts/search_step1_generate.sh
```

Step 1 training:

```bash
!GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_train.sh
```

Step 1 evaluation and statistics:

```bash
!GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_eval.sh
!IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_stat.sh
```

Step 2 generation from the chosen template:

```bash
!TEMPLATE=10 bash scripts/search_step2_generate.sh
```

Step 2 training, evaluation, and statistics:

```bash
!GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_train.sh
!GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_eval.sh
!IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_stat.sh
```

## 6. Save outputs and resume work

- Checkpoints land in `SCRFD_WORK_ROOT` and default to `work_dirs/`.
- WIDER FACE evaluation outputs land in `SCRFD_RESULT_ROOT` and default to `wouts/`.
- Kaggle persists `/kaggle/working` for the current session only, so copy important checkpoints/results to a Kaggle Dataset or external storage before ending the session.
- To resume a baseline run, pass `EXTRA_TRAIN_ARGS="--resume-from /path/to/checkpoint.pth"` to `scripts/train_baseline.sh` or `scripts/train_scrfd_1g.sh`.

Example:

```bash
!EXTRA_TRAIN_ARGS="--resume-from /kaggle/working/scrfd-fullsearch-kaggle/work_dirs/scrfd_1g/latest.pth" \
  bash scripts/train_scrfd_1g.sh
```

## 7. Practical Kaggle tips

- Keep `workers_per_gpu` small if notebook RAM is tight.
- Search runs are much longer than a single baseline sanity check; persist intermediate checkpoints often.
- If Kaggle storage is tight, move finished candidate folders out of `work_dirs/` after recording their metrics.
- The upstream public release exposes the original two-step search flow around the `2.5G` example; this mirror preserves that path rather than inventing a new search framework.
