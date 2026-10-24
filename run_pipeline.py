"""
run_pipeline.py
===============
Master Pipeline Runner for Lake Bloom Severity Prediction (lake_bloom_xgboost).
Executes the pipeline stages in sequence:
  1. Preprocessing (raw -> processed clean_data.csv)
  2. Hyperparameter Tuning (GroupKFold CV -> configs/best_params.yaml)
  3. Final Training (outputs/models/xgb_model.pkl)
  4. Evaluation & Reporting (outputs/reports/)
"""

import os
import sys
import time
import argparse
import subprocess

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_command(cmd, desc):
    """Executes a python sub-stage script and checks return code."""
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {desc.upper()}")
    print("=" * 75)
    start_time = time.time()

    # Pass current environment and run with python executable
    proc = subprocess.run([sys.executable] + cmd, env=os.environ.copy())
    elapsed = time.time() - start_time

    if proc.returncode != 0:
        print(f"\n[ERROR] Step '{desc}' failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    print(f"[SUCCESS] {desc} finished in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Run the lake_bloom_xgboost ML pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "finetune", "train", "evaluate"],
        default="all",
        help="Specify which stage to run (default: all)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter search and use existing or base parameters",
    )
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("=" * 75)
    print("    LAKE BLOOM SEVERITY PREDICTION - XGBOOST PIPELINE RUNNER")
    print("=" * 75)
    print(f"Working Directory: {project_dir}")
    print(f"Selected Stage   : {args.stage}")
    print(f"Skip Tuning      : {args.skip_tuning}")

    total_start = time.time()

    # Stage 1: Preprocessing
    if args.stage in ["all", "preprocess"]:
        run_command(["preprocessing/preprocess.py"], "1. Data Preprocessing (raw -> processed)")

    # Stage 2: Finetuning
    if args.stage in ["all", "finetune"] and not args.skip_tuning:
        run_command(["model/finetune.py"], "2. Hyperparameter Tuning (GroupKFold CV)")

    # Stage 3: Training
    if args.stage in ["all", "train"]:
        run_command(["model/train.py"], "3. Final Model Training")

    # Stage 4: Evaluation
    if args.stage in ["all", "evaluate"]:
        run_command(["model/evaluate.py"], "4. Evaluation, Reporting & Visualization")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s")
    print("=" * 75)
    print(f"  • Trained model : outputs/models/xgb_model.pkl")
    print(f"  • Reports & plots: outputs/reports/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
