# Kaggle Quickstart: SCRFD 2.5G + Online SR

## Cell 1: Clone repo
```bash
!git clone https://github.com/TQC0103/scrfd-fullsearch-kaggle.git
%cd /kaggle/working/scrfd-fullsearch-kaggle
!git pull origin main
```

## Cell 2: Prepare environment
```bash
!bash scripts/prepare_env.sh
```

If your Kaggle notebook already uses the packed environment flow in this repo, use that flow instead of re-running full setup.

## Cell 3: Prepare dataset paths
Make sure your WIDER FACE data is mounted in Kaggle under the paths expected by the repo:

```text
/kaggle/input/<your-widerface-dataset>/
```

If needed, export custom paths before training:

```bash
%env SCRFD_DATA_ROOT=/kaggle/input/<your-widerface-dataset>
```

## Cell 4: Quick sanity run
```bash
!SCRFD_TOTAL_EPOCHS=16 SCRFD_EVAL_INTERVAL=4 SCRFD_CHECKPOINT_INTERVAL=4 bash scripts/train_scrfd_2.5g_online_sr.sh
```

## Cell 5: Full training
```bash
!bash scripts/train_scrfd_2.5g_online_sr.sh
```

## Cell 6: Watch scheduler state
```bash
!cat work_dirs/scrfd_2.5g_online_sr/sr_scheduler_state.json
```

## What this runs
- Fixed public `SCRFD-2.5G` architecture
- Online sample redistribution only
- Dynamic `RandomSquareCrop` probabilities updated once per epoch from stride-wise training stats

## Main files
- `configs/scrfd/scrfd_2.5g_online_sr.py`
- `scripts/train_scrfd_2.5g_online_sr.sh`
- `mmdet/core/utils/online_sr_hook.py`
