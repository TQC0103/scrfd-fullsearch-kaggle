#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw WIDER FACE files into a Kaggle-ready SCRFD dataset zip."
    )
    parser.add_argument(
        "--wider-root",
        type=Path,
        required=True,
        help="Root folder that contains WIDER_train, WIDER_val, wider_face_split, and eval_tools.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        required=True,
        help="Output zip path, e.g. retinaface-kaggle-upload.zip",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also include WIDER_test/images in the zip under widerface_test/ for reference.",
    )
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def convert_split(split_txt: Path, image_root: Path) -> tuple[str, dict[str, int]]:
    ensure_exists(split_txt)
    ensure_exists(image_root)

    lines: list[str] = []
    stats = {"images": 0, "boxes": 0, "ignored_boxes": 0, "empty_images": 0}

    with open(split_txt, "r", encoding="utf-8") as handle:
        state = "path"
        current_rel = ""
        remaining = 0
        for raw in handle:
            line = raw.strip()
            if state == "path":
                current_rel = line
                state = "count"
                continue

            if state == "count":
                remaining = int(line)
                image_path = image_root / current_rel
                ensure_exists(image_path)
                width, height = read_image_size(image_path)
                lines.append(f"# {current_rel} {width} {height}")
                stats["images"] += 1
                if remaining == 0:
                    stats["empty_images"] += 1
                    state = "skip_zero_placeholder"
                else:
                    state = "box"
                continue

            if state == "skip_zero_placeholder":
                state = "path"
                continue

            values = [int(float(x)) for x in line.split()]
            if len(values) < 10:
                raise ValueError(f"Unexpected annotation line: {line}")

            x1, y1, w, h = values[:4]
            invalid = values[7]
            if w <= 0 or h <= 0:
                remaining -= 1
                if remaining == 0:
                    state = "path"
                continue

            x2 = x1 + w
            y2 = y1 + h
            ignore = 1 if invalid == 1 else 0

            lines.append(f"{x1} {y1} {x2} {y2} {ignore}")
            stats["boxes"] += 1
            stats["ignored_boxes"] += ignore

            remaining -= 1
            if remaining == 0:
                state = "path"

    return "\n".join(lines) + "\n", stats


def add_tree_to_zip(zip_file: ZipFile, src_root: Path, dst_root: str, suffix: str = ".jpg") -> int:
    count = 0
    for path in src_root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        arcname = f"{dst_root}/{path.relative_to(src_root).as_posix()}"
        zip_file.write(path, arcname=arcname)
        count += 1
    return count


def add_file(zip_file: ZipFile, src: Path, dst: str) -> None:
    ensure_exists(src)
    zip_file.write(src, arcname=dst)


def main() -> None:
    args = parse_args()

    wider_root = args.wider_root
    train_images = wider_root / "WIDER_train" / "images"
    val_images = wider_root / "WIDER_val" / "images"
    test_images = wider_root / "WIDER_test" / "images"
    split_root = wider_root / "wider_face_split"
    gt_root = wider_root / "eval_tools" / "eval_tools" / "ground_truth"

    for required in [train_images, val_images, split_root, gt_root]:
        ensure_exists(required)

    train_label, train_stats = convert_split(
        split_root / "wider_face_train_bbx_gt.txt", train_images
    )
    val_label, val_stats = convert_split(
        split_root / "wider_face_val_bbx_gt.txt", val_images
    )

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(wider_root),
        "train": train_stats,
        "val": val_stats,
        "include_test": bool(args.include_test),
    }

    with ZipFile(args.output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zip_file:
        zip_file.writestr("retinaface/train/labelv2.txt", train_label)
        zip_file.writestr("retinaface/val/labelv2.txt", val_label)

        add_tree_to_zip(zip_file, train_images, "retinaface/train/images")
        add_tree_to_zip(zip_file, val_images, "retinaface/val/images")

        for name in [
            "wider_face_val.mat",
            "wider_easy_val.mat",
            "wider_medium_val.mat",
            "wider_hard_val.mat",
        ]:
            add_file(zip_file, gt_root / name, f"retinaface/val/gt/{name}")

        if args.include_test and test_images.exists():
            test_count = add_tree_to_zip(zip_file, test_images, "retinaface/test/images")
            add_file(
                zip_file,
                split_root / "wider_face_test_filelist.txt",
                "retinaface/test/wider_face_test_filelist.txt",
            )
            manifest["test"] = {"images": test_count}

        zip_file.writestr(
            "retinaface/MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        )

    print(json.dumps({"output_zip": str(args.output_zip), **manifest}, indent=2))


if __name__ == "__main__":
    main()
