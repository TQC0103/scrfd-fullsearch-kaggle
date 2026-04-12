import argparse
import shutil
import zipfile
from pathlib import Path


EXCLUDE_NAMES = {
    ".git",
    "work_dirs",
    "wouts",
    "outputs",
    "logs",
    "tmp",
    "dist",
    "__pycache__",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Kaggle-safe offline SCRFD source bundle")
    parser.add_argument("--repo-root", default=None, help="repository root; defaults to parent of this script")
    parser.add_argument("--output-dir", default="dist", help="directory where the zip bundle will be written")
    parser.add_argument("--bundle-name", default="scrfd-fullsearch-kaggle-offline.zip", help="zip filename")
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    bundle_path = output_dir / args.bundle_name
    repo_name = repo_root.name

    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in repo_root.rglob("*"):
            if path.is_dir():
                continue
            rel_path = path.relative_to(repo_root)
            if should_skip(rel_path):
                continue
            arcname = Path(repo_name, rel_path).as_posix()
            zf.write(path, arcname)

    print(f"Created offline bundle: {bundle_path}")


if __name__ == "__main__":
    main()
