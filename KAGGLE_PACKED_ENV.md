# Kaggle Packed Environment

Use this path when you want to avoid rebuilding the SCRFD runtime inside a Kaggle notebook.

## What to upload

Upload these as Kaggle Datasets:

1. `scrfd-fullsearch-kaggle-offline.zip`
2. `retinaface-kaggle-upload-with-test.zip`
3. `scrfd-kaggle-baseline-py312-cu128.tar.gz`

The packed environment archive is created from Linux/WSL and already includes:

- Python 3.12
- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `mmcv==1.4.0`
- baseline runtime dependencies needed for SCRFD

## Restore inside Kaggle

After attaching the packed environment dataset, restore it with:

```bash
bash scripts/restore_kaggle_packed_env.sh \
  /kaggle/input/<packed-env-dataset>/scrfd-kaggle-baseline-py312-cu128.tar.gz \
  /kaggle/working/.scrfd-packed-env
```

The script prints the Python path for the restored environment:

```text
/kaggle/working/.scrfd-packed-env/bin/python
```

## Run baseline train

```bash
PYTHONPATH=/kaggle/working/scrfd-fullsearch-kaggle \
/kaggle/working/.scrfd-packed-env/bin/python \
  /kaggle/working/scrfd-fullsearch-kaggle/scripts/kaggle_competition_entry.py \
  --mode baseline_train_quick \
  --data-root /kaggle/input/<widerface-dataset>/retinaface \
  --work-root /kaggle/working/scrfd-fullsearch-kaggle/work_dirs \
  --result-root /kaggle/working/scrfd-fullsearch-kaggle/wouts \
  --gpu-id 0 \
  --total-epochs 16 \
  --eval-interval 4 \
  --checkpoint-interval 4
```

## Build the archive from WSL

Use:

```bash
bash scripts/build_wsl_kaggle_packed_env.sh
```

That script writes the archive to:

```text
dist/upload_ready/scrfd-kaggle-baseline-py312-cu128.tar.gz
```
