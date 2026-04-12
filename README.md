# SCRFD Full Search Kaggle Mirror

This repository is a clean public mirror of the SCRFD subtree from the official [InsightFace](https://github.com/deepinsight/insightface) repository, extracted from `detection/scrfd` and reorganized for a simpler research workflow on Kaggle.

The main goal is **not** to provide only a baseline face detector. The main goal is to **preserve the full SCRFD two-step NAS/search workflow** so that you can:

1. rerun the original search path,
2. inspect and modify the search pipeline,
3. compare original vs. improved search logic,
4. clone a single public repository on Kaggle and keep working from there.

Baseline training and evaluation are still included, but only as a sanity-check path.

## Scope

- Primary workflow: full SCRFD search.
- Sanity-check workflow: baseline train/eval, especially `SCRFD-1.0GF`.
- Design goal: keep the upstream search path intact while making the repo easier to clone, configure, and edit on Kaggle.

## What Was Preserved From Upstream

This mirror keeps the parts needed to reproduce the original SCRFD public workflow:

- `configs/` for baseline models and generated search candidates.
- `search_tools/` for candidate generation, candidate training orchestration, evaluation orchestration, and statistics collection.
- `mmdet/` with the SCRFD detector, head, dataset, transforms, losses, and backbone code required by both baseline and search.
- `tools/` for training, testing, WIDER FACE evaluation, FLOPs calculation, and export utilities.
- `setup.py`, `requirements/`, and local package layout so the repo can still be installed and run as a standalone project.

## What Changed In This Mirror

- Added `scripts/` wrappers with clearer entry points for baseline sanity checks and step-by-step search runs.
- Added env/CLI-friendly dataset and output handling so Kaggle notebooks do not need personal hard-coded paths.
- Patched evaluation/statistics helpers so they accept config overrides and explicit ranges/paths.
- Removed the demo folder to keep the mirror focused on training, evaluation, and search.
- Rewrote the README and added `KAGGLE.md` for practical usage.

## Repository Structure

```text
configs/             baseline configs + search seed/template configs
mmdet/               detector, head, dataset, transforms, losses, backbones
requirements/        upstream dependency groups
scripts/             Kaggle-friendly wrapper entry points
search_tools/        original search generation/train/test/stat utilities
tools/               train/test/eval/export helpers
KAGGLE.md            Kaggle-specific notes
README.md            repo overview + workflow guide
```

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/TQC0103/scrfd-fullsearch-kaggle.git
   cd scrfd-fullsearch-kaggle
   ```

2. Install the Python dependencies kept in this mirror:

   ```bash
   bash scripts/prepare_env.sh
   ```

3. Install a version of `mmcv-full` that matches the active PyTorch/CUDA environment. The original upstream README mentioned `mmcv-full==1.2.6` and `1.3.3` as tested combinations, but Kaggle images change over time, so keep this step explicit.

4. Confirm the local package is installed in editable mode:

   ```bash
   python -c "import mmdet; print(mmdet.__version__)"
   ```

## Dataset Preparation

The wrapper scripts read WIDER FACE from `SCRFD_DATA_ROOT`.

Expected layout:

```text
$SCRFD_DATA_ROOT/
  train/
    images/
    labelv2.txt
  val/
    images/
    labelv2.txt
    gt/
      *.mat
```

Example on Kaggle:

```bash
export SCRFD_DATA_ROOT=/kaggle/input/widerface-retinaface-format/retinaface
export SCRFD_WORK_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/work_dirs
export SCRFD_RESULT_ROOT=/kaggle/working/scrfd-fullsearch-kaggle/wouts
```

The raw upstream configs still default to `data/retinaface`. The new wrapper scripts override those paths via `--cfg-options`, so you do not have to edit config files just to move between local and Kaggle environments.

## Quick Sanity Check

Train `SCRFD-1.0GF` quickly for Kaggle sanity checks:

```bash
bash scripts/train_scrfd_1g_quick.sh
```

This wrapper keeps the model at `SCRFD-1.0GF` but drops training to a short Kaggle-friendly run by default (`16` epochs, eval/checkpoint every `4` epochs). If you explicitly want the longer upstream-style baseline schedule, use:

```bash
bash scripts/train_scrfd_1g.sh
```

Evaluate `SCRFD-1.0GF` on WIDER FACE:

```bash
CHECKPOINT=work_dirs/scrfd_1g/latest.pth bash scripts/eval_scrfd_1g.sh
```

You can also use the generic baseline wrappers for other bundled configs:

```bash
CONFIG_NAME=scrfd_500m bash scripts/train_baseline.sh
CONFIG_NAME=scrfd_2.5g bash scripts/eval_baseline.sh
```

## Full Two-Step Search Workflow

The original public SCRFD release exposes the search pipeline around the `2.5G` example. This mirror preserves that path and adds cleaner entry points.

### Step 1: backbone search

Generate step-1 candidates:

```bash
bash scripts/search_step1_generate.sh
```

Train candidates:

```bash
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_train.sh
```

Evaluate candidates:

```bash
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_eval.sh
```

Summarize candidate statistics:

```bash
IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_stat.sh
```

### Step 2: detector / full-network search

Choose a template from the step-1 ranking, then generate step-2 candidates:

```bash
TEMPLATE=10 bash scripts/search_step2_generate.sh
```

Train step-2 candidates:

```bash
GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_train.sh
```

Evaluate step-2 candidates:

```bash
GPU_ID=0 IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_eval.sh
```

Summarize step-2 statistics:

```bash
IDX_FROM=65 IDX_TO=128 bash scripts/search_step2_stat.sh
```

### Optional wrapper

For long-running environments where you intentionally want to chain the full example pipeline:

```bash
bash scripts/run_full_search_pipeline.sh
```

## Where To Modify The Search Algorithm

If you want to research search improvements, these are the main files to edit:

- `search_tools/generate_configs_2.5g.py`
  - search space definition
  - candidate generation logic
  - FLOPs filtering
  - step-1 vs. step-2 generation behavior
- `search_tools/search_stat.py`
  - result aggregation
  - ranking logic
  - export format for downstream selection
- `search_tools/search_train.py`
  - candidate training orchestration
  - launcher strategy
  - per-candidate config overrides
- `configs/scrfdgen2.5g/scrfdgen2.5g_0.py`
  - seed/template config for the public `2.5G` search example
- `mmdet/models/backbones/`
  - backbone search space implementation details
- `mmdet/models/dense_heads/scrfd_head.py`
  - detector head design and behavior
- `mmdet/models/detectors/scrfd.py`
  - detector-level logic

## Kaggle Notes

- Clone into `/kaggle/working` and keep outputs in `/kaggle/working/...` so the notebook session can access them directly.
- Point `SCRFD_DATA_ROOT` to a Kaggle Dataset mount rather than editing configs.
- Default outputs:
  - checkpoints/logs: `SCRFD_WORK_ROOT`, default `work_dirs/`
  - WIDER FACE results: `SCRFD_RESULT_ROOT`, default `wouts/`
- Resume by passing `EXTRA_TRAIN_ARGS="--resume-from /path/to/latest.pth"` to the baseline wrapper or `--resume-from` to `search_tools/search_train.py`.
- Search runs are long for a single Kaggle GPU; persist intermediate checkpoints/results before the session ends.
- If workers cause memory pressure on Kaggle, reduce `workers_per_gpu` in the relevant config.

More notebook-oriented examples are in [KAGGLE.md](KAGGLE.md).
If you are using a Kaggle **competition** notebook with internet disabled, use [KAGGLE_OFFLINE.md](KAGGLE_OFFLINE.md) instead.
The repository also includes a ready-made notebook template at `notebooks/kaggle_competition_click_run.ipynb`.

## Main Wrapper Scripts

- `scripts/prepare_env.sh`
- `scripts/package_offline_bundle.ps1`
- `scripts/package_offline_bundle.sh`
- `scripts/train_baseline.sh`
- `scripts/eval_baseline.sh`
- `scripts/train_scrfd_1g.sh`
- `scripts/eval_scrfd_1g.sh`
- `scripts/search_step1_generate.sh`
- `scripts/search_step1_train.sh`
- `scripts/search_step1_eval.sh`
- `scripts/search_step1_stat.sh`
- `scripts/search_step2_generate.sh`
- `scripts/search_step2_train.sh`
- `scripts/search_step2_eval.sh`
- `scripts/search_step2_stat.sh`
- `scripts/run_full_search_pipeline.sh`

## Known Caveats

- This mirror preserves the public SCRFD search release path from InsightFace. That path is still centered around the published `2.5G` example rather than a generalized search framework for every FLOPs budget.
- Kaggle compatibility is improved through wrappers and path overrides, but `mmcv-full` remains the dependency most sensitive to the active Kaggle image.
- Search remains computationally expensive. This repo makes it easier to run and modify, but it does not make long NAS experiments cheap.

## Source

- Source repository: `https://github.com/deepinsight/insightface`
- Source subtree: `detection/scrfd`
