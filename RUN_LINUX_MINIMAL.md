# Minimal Linux Setup

If you want the fewest manual steps on a Linux GPU machine, use one of these:

## Option 1: Conda

```bash
git clone https://github.com/TQC0103/scrfd-fullsearch-kaggle.git
cd scrfd-fullsearch-kaggle
bash scripts/bootstrap_linux_env.sh
conda activate scrfd
export SCRFD_DATA_ROOT=/path/to/retinaface
export SCRFD_WORK_ROOT=$PWD/work_dirs
export SCRFD_RESULT_ROOT=$PWD/wouts
bash scripts/train_scrfd_1g_quick.sh
```

## Option 2: Docker

```bash
git clone https://github.com/TQC0103/scrfd-fullsearch-kaggle.git
cd scrfd-fullsearch-kaggle
docker build -t scrfd-fullsearch .
docker run --gpus all -it --rm \
  -v /path/to/retinaface:/data/retinaface \
  -v $PWD:/workspace/scrfd-fullsearch-kaggle \
  scrfd-fullsearch bash
```

Inside the container:

```bash
export SCRFD_DATA_ROOT=/data/retinaface
export SCRFD_WORK_ROOT=/workspace/scrfd-fullsearch-kaggle/work_dirs
export SCRFD_RESULT_ROOT=/workspace/scrfd-fullsearch-kaggle/wouts
bash scripts/train_scrfd_1g_quick.sh
```

## Search

After baseline sanity check:

```bash
bash scripts/search_step1_generate.sh
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_train.sh
GPU_ID=0 IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_eval.sh
IDX_FROM=1 IDX_TO=64 bash scripts/search_step1_stat.sh
```

