import argparse
import json
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = [
    "scrfd-fullsearch-kaggle/README.md",
    "scrfd-fullsearch-kaggle/KAGGLE.md",
    "scrfd-fullsearch-kaggle/KAGGLE_OFFLINE.md",
    "scrfd-fullsearch-kaggle/requirements.txt",
    "scrfd-fullsearch-kaggle/configs/scrfd/scrfd_1g.py",
    "scrfd-fullsearch-kaggle/configs/scrfdgen2.5g/scrfdgen2.5g_0.py",
    "scrfd-fullsearch-kaggle/scripts/common.sh",
    "scrfd-fullsearch-kaggle/scripts/train_scrfd_1g.sh",
    "scrfd-fullsearch-kaggle/scripts/eval_scrfd_1g.sh",
    "scrfd-fullsearch-kaggle/scripts/search_step1_generate.sh",
    "scrfd-fullsearch-kaggle/scripts/search_step2_generate.sh",
    "scrfd-fullsearch-kaggle/scripts/kaggle_competition_entry.py",
    "scrfd-fullsearch-kaggle/search_tools/generate_configs_2.5g.py",
    "scrfd-fullsearch-kaggle/search_tools/search_train.py",
    "scrfd-fullsearch-kaggle/search_tools/search_stat.py",
    "scrfd-fullsearch-kaggle/tools/train.py",
    "scrfd-fullsearch-kaggle/tools/test_widerface.py",
    "scrfd-fullsearch-kaggle/notebooks/kaggle_competition_click_run.ipynb",
]


FORBIDDEN_PREFIXES = [
    "scrfd-fullsearch-kaggle/.git/",
    "scrfd-fullsearch-kaggle/dist/",
    "scrfd-fullsearch-kaggle/work_dirs/",
    "scrfd-fullsearch-kaggle/wouts/",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Kaggle offline bundle contents")
    parser.add_argument(
        "--zip-path",
        default="dist/scrfd-fullsearch-kaggle-offline.zip",
        help="path to the offline source bundle",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the verification result as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"Missing bundle: {zip_path}", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    missing = [name for name in REQUIRED_FILES if name not in names]
    forbidden = []
    for name in names:
        if "\\" in name:
            forbidden.append(name)
            continue
        for prefix in FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                forbidden.append(name)
                break

    result = {
        "zip_path": str(zip_path.resolve()),
        "size_bytes": zip_path.stat().st_size,
        "required_count": len(REQUIRED_FILES),
        "missing_required": missing,
        "forbidden_entries": forbidden,
        "ready": not missing and not forbidden,
        "external_items_still_needed": [
            "A Kaggle Dataset containing this zip file",
            "A Kaggle Dataset containing the WIDER FACE / retinaface-format dataset",
            "Optional wheelhouse dataset if the competition image is missing dependencies such as mmcv/mmcv-full",
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Offline bundle verification")
        print("zip_path:", result["zip_path"])
        print("size_bytes:", result["size_bytes"])
        print("required_count:", result["required_count"])
        print("missing_required:", len(missing))
        for item in missing:
            print("  MISSING:", item)
        print("forbidden_entries:", len(forbidden))
        for item in forbidden:
            print("  FORBIDDEN:", item)
        print("ready:", result["ready"])
        print("external_items_still_needed:")
        for item in result["external_items_still_needed"]:
            print("  -", item)

    sys.exit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
