"""Project-root helpers."""

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(rel_path: str) -> dict:
    path = project_root() / rel_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config() -> dict:
    return load_yaml("configs/config.yaml")


def load_evaluation_config() -> dict:
    return load_yaml("configs/evaluation.yaml")


def load_model_config(model_name: str) -> dict:
    return load_yaml(f"configs/models/{model_name}.yaml")


KNOWN_MODELS = (
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
)
