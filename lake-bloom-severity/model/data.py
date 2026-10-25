"""Load processed data, verify checksum, apply the frozen location split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from model.paths import load_config, project_root


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series
    feature_cols: list[str]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_processed_frame(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = project_root()
    path = root / cfg["paths"]["processed_data"]
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")

    expected = cfg["data"].get("checksum_sha256")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}.\nexpected: {expected}\nactual:   {actual}\n"
            "Update configs/config.yaml if the processed file change is intentional."
        )
    return pd.read_csv(path)


def load_frozen_split(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = project_root() / cfg["paths"]["splits"]
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen split not found: {path}. Run python preprocessing/make_split.py"
        )
    return pd.read_csv(path)


def get_feature_frame(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cfg = cfg or load_config()
    target = cfg["data"]["target_col"]
    group = cfg["data"]["group_col"]
    feature_cols = list(cfg["data"]["feature_cols"])
    forbidden = set(cfg["data"]["forbidden_cols"])

    overlap = set(feature_cols) & forbidden
    if overlap:
        raise ValueError(f"Feature list includes forbidden columns: {sorted(overlap)}")

    missing = [c for c in feature_cols + [target, group] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processed data: {missing}")

    X = df[feature_cols].copy()
    y = df[target].astype(int)
    groups = df[group]
    return X, y, groups


def load_train_test(cfg: dict | None = None) -> SplitData:
    cfg = cfg or load_config()
    df = load_processed_frame(cfg)
    splits = load_frozen_split(cfg)
    group = cfg["data"]["group_col"]

    merged = df.merge(splits, on=group, how="left")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), group].nunique()
        raise ValueError(f"{missing} location groups are not in splits.csv")

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]
    overlap = set(train[group].unique()) & set(test[group].unique())
    if overlap:
        raise ValueError(f"CRITICAL: {len(overlap)} location groups leaked between train and test")

    X_train, y_train, g_train = get_feature_frame(train, cfg)
    X_test, y_test, g_test = get_feature_frame(test, cfg)
    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        groups_train=g_train.reset_index(drop=True),
        groups_test=g_test.reset_index(drop=True),
        feature_cols=list(cfg["data"]["feature_cols"]),
    )
