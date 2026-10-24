"""
model/utils.py
==============
Shared utilities: configuration loading, location-based GroupShuffleSplit,
one-hot encoding pipeline builders, label mappings, and model serialization.
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight


def get_base_dir():
    """Returns the base directory of the lake_bloom_xgboost package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_rel_path="configs/config.yaml"):
    """Loads a YAML configuration file relative to the package base directory."""
    base_dir = get_base_dir()
    full_path = os.path.join(base_dir, config_rel_path)
    if not os.path.exists(full_path):
        # Fallback if config_rel_path is already absolute
        full_path = config_rel_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(config=None):
    """Loads the processed clean dataset."""
    if config is None:
        config = load_config()
    base_dir = get_base_dir()
    data_path = os.path.join(base_dir, config["paths"]["processed_data"])
    if not os.path.exists(data_path):
        # Fallback to current working directory or backup
        backup = os.path.join(base_dir, "..", "clean_data.csv")
        if os.path.exists(backup):
            data_path = backup
    df = pd.read_csv(data_path)
    return df


def get_train_test_split(df, config=None, exclude_feature=None):
    """
    Performs location-based GroupShuffleSplit ensuring zero location leakage.
    Maps target classes {1, 2, 3, 4} to {0, 1, 2, 3} for XGBoost multi:softprob.
    """
    if config is None:
        config = load_config()

    target_col = config["data"]["target_col"]
    group_col = config["data"]["group_col"]
    forbidden_cols = set(config["data"]["forbidden_cols"])

    feature_cols = [c for c in df.columns if c not in forbidden_cols]
    if exclude_feature and exclude_feature in feature_cols:
        feature_cols.remove(exclude_feature)

    X = df[feature_cols]
    y_raw = df[target_col]
    groups = df[group_col]

    # Map labels 1..4 -> 0..3 for XGBoost compatibility
    unique_classes = sorted(y_raw.unique())
    label_map = {orig: idx for idx, orig in enumerate(unique_classes)}
    inverse_label_map = {idx: orig for orig, idx in label_map.items()}
    y = y_raw.map(label_map)

    # Perform GroupShuffleSplit
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Leakage assertion check
    overlap = set(groups_train.unique()) & set(groups_test.unique())
    if len(overlap) != 0:
        raise ValueError(f"CRITICAL ERROR: {len(overlap)} location groups leaked between train and test!")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_cols,
        label_map,
        inverse_label_map,
    )


def build_preprocessor(cat_cols, num_cols):
    """Builds a ColumnTransformer with OneHotEncoder for categoricals and passthrough for numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def compute_balanced_weights(y_train):
    """Computes sample weights inversely proportional to class frequencies to handle target imbalance."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


def save_artifact(obj, rel_path):
    """Saves a python object using joblib to a path relative to project root."""
    base_dir = get_base_dir()
    full_path = os.path.join(base_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(obj, full_path)
    print(f"Artifact saved: {full_path}")


def load_artifact(rel_path):
    """Loads a python object from joblib relative to project root."""
    base_dir = get_base_dir()
    full_path = os.path.join(base_dir, rel_path)
    return joblib.load(full_path)
