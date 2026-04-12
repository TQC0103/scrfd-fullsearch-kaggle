import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "torch",
    "cv2",
    "mmcv",
    "autotorch",
    "terminaltables",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Kaggle competition entry point for SCRFD baseline/search runs."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "baseline_train",
            "baseline_eval",
            "step1_generate",
            "step1_train",
            "step1_eval",
            "step1_stat",
            "step2_generate",
            "step2_train",
            "step2_eval",
            "step2_stat",
            "full_search",
        ],
    )
    parser.add_argument("--data-root", required=True, help="WIDER FACE / retinaface-format dataset root")
    parser.add_argument("--work-root", default=None, help="Checkpoint/output work root")
    parser.add_argument("--result-root", default=None, help="WIDER eval result root")
    parser.add_argument("--wheelhouse", default=None, help="Offline wheel directory dataset")
    parser.add_argument("--skip-offline-install", action="store_true", help="Skip offline pip installation")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--idx-from", type=int, default=None)
    parser.add_argument("--idx-to", type=int, default=None)
    parser.add_argument("--template", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thr", default=None)
    parser.add_argument("--mode-value", default=None, help="Forwarded to eval wrappers as MODE")
    parser.add_argument("--config-name", default=None, help="Optional baseline config override")
    parser.add_argument("--config-group", default=None, help="Optional baseline config group override")
    return parser.parse_args()


def has_module(name):
    return importlib.util.find_spec(name) is not None


def ensure_pythonpath(repo_root):
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    current = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = repo_str if not current else repo_str + ":" + current


def install_offline_dependencies(repo_root, wheelhouse):
    missing = [name for name in REQUIRED_IMPORTS if not has_module(name)]
    if not missing:
        print("All required imports already available.")
        return

    if not wheelhouse or not Path(wheelhouse).exists():
        raise RuntimeError(
            "Missing packages detected: %s. Provide --wheelhouse pointing to a Kaggle Dataset "
            "with offline wheels, or preinstall them in the competition image." % ", ".join(missing)
        )

    print("Installing missing/offline dependencies from:", wheelhouse)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        wheelhouse,
        "-r",
        str(repo_root / "requirements.txt"),
    ]
    subprocess.run(cmd, check=True)

    remaining = [name for name in REQUIRED_IMPORTS if not has_module(name)]
    if remaining:
        raise RuntimeError(
            "Offline install finished but these modules are still missing: %s. "
            "Your wheelhouse likely needs more wheels, especially mmcv/mmcv-full."
            % ", ".join(remaining)
        )


def build_env(args, repo_root):
    env = os.environ.copy()
    env["SCRFD_REPO_ROOT"] = str(repo_root)
    env["SCRFD_DATA_ROOT"] = args.data_root
    env["SCRFD_WORK_ROOT"] = args.work_root or str(repo_root / "work_dirs")
    env["SCRFD_RESULT_ROOT"] = args.result_root or str(repo_root / "wouts")
    env["GPU_ID"] = str(args.gpu_id)

    if args.idx_from is not None:
        env["IDX_FROM"] = str(args.idx_from)
    if args.idx_to is not None:
        env["IDX_TO"] = str(args.idx_to)
    if args.template is not None:
        env["TEMPLATE"] = str(args.template)
    if args.checkpoint:
        env["CHECKPOINT"] = args.checkpoint
    if args.thr:
        env["THR"] = str(args.thr)
    if args.mode_value:
        env["MODE"] = str(args.mode_value)
    if args.config_name:
        env["CONFIG_NAME"] = args.config_name
    if args.config_group:
        env["CONFIG_GROUP"] = args.config_group

    return env


def dispatch_script(mode):
    mapping = {
        "baseline_train": "scripts/train_scrfd_1g.sh",
        "baseline_eval": "scripts/eval_scrfd_1g.sh",
        "step1_generate": "scripts/search_step1_generate.sh",
        "step1_train": "scripts/search_step1_train.sh",
        "step1_eval": "scripts/search_step1_eval.sh",
        "step1_stat": "scripts/search_step1_stat.sh",
        "step2_generate": "scripts/search_step2_generate.sh",
        "step2_train": "scripts/search_step2_train.sh",
        "step2_eval": "scripts/search_step2_eval.sh",
        "step2_stat": "scripts/search_step2_stat.sh",
        "full_search": "scripts/run_full_search_pipeline.sh",
    }
    return mapping[mode]


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ensure_pythonpath(repo_root)

    if not args.skip_offline_install:
        install_offline_dependencies(repo_root, args.wheelhouse)

    env = build_env(args, repo_root)
    script_path = repo_root / dispatch_script(args.mode)

    print("Running mode:", args.mode)
    print("Repo root:", repo_root)
    print("Data root:", env["SCRFD_DATA_ROOT"])
    print("Work root:", env["SCRFD_WORK_ROOT"])
    print("Result root:", env["SCRFD_RESULT_ROOT"])

    subprocess.run(["bash", str(script_path)], cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()
