"""Create frozen location-level train/test assignment (run once, then commit)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.paths import load_config, project_root


def make_split(processed_path: Path, splits_path: Path, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(processed_path)
    target = cfg["data"]["target_col"]
    group = cfg["data"]["group_col"]
    split_cfg = cfg["split"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    train_idx, test_idx = next(gss.split(df, df[target], df[group]))
    train_groups = set(df.iloc[train_idx][group].unique())
    test_groups = set(df.iloc[test_idx][group].unique())
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"{len(overlap)} location groups leaked between train and test")

    rows = [{"location_group": g, "split": "train"} for g in sorted(train_groups)]
    rows += [{"location_group": g, "split": "test"} for g in sorted(test_groups)]
    splits = pd.DataFrame(rows)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(splits_path, index=False)

    print(
        f"wrote {splits_path}: {len(train_groups)} train locations, "
        f"{len(test_groups)} test locations, overlap={len(overlap)}"
    )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Write frozen splits.csv by location_group")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing splits.csv (avoid unless you intend to change the split)",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()
    processed = root / cfg["paths"]["processed_data"]
    splits_path = root / cfg["paths"]["splits"]
    if splits_path.exists() and not args.force:
        print(f"{splits_path} already exists. Pass --force to overwrite.")
        return
    make_split(processed, splits_path, cfg)


if __name__ == "__main__":
    main()
