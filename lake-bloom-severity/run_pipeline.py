"""Run pipeline stages for any list of models."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.paths import KNOWN_MODELS


def run_step(args: list[str], desc: str) -> None:
    print("\n" + "=" * 70)
    print(desc)
    print("=" * 70)
    start = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed ({desc}) with code {proc.returncode}")
    print(f"[ok] {desc} in {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lake bloom severity pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(KNOWN_MODELS),
        choices=KNOWN_MODELS,
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "split", "finetune", "train", "evaluate", "compare"],
    )
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    args = parser.parse_args()

    if args.stage in ("all", "preprocess"):
        run_step(["preprocessing/preprocess.py"], "preprocess")

    if args.stage in ("all", "split"):
        split_cmd = ["preprocessing/make_split.py"]
        if args.force_split:
            split_cmd.append("--force")
        run_step(split_cmd, "make_split")

    do_tune = args.stage in ("all", "finetune") and not args.skip_tuning
    do_train = args.stage in ("all", "train")
    do_eval = args.stage in ("all", "evaluate")

    for model_name in args.models:
        if do_tune:
            run_step(["model/finetune.py", "--model", model_name], f"finetune {model_name}")
        if do_train:
            run_step(["model/train.py", "--model", model_name], f"train {model_name}")
        if do_eval:
            run_step(["model/evaluate.py", "--model", model_name], f"evaluate {model_name}")

    if args.stage in ("all", "compare"):
        run_step(["model/compare.py"], "compare")


if __name__ == "__main__":
    main()
