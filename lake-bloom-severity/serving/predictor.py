"""One predict function per model, plus a dispatcher for the backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from serving.schema import SchemaError, load_schema, validate_payload

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CLASSES = (1, 2, 3, 4)


class PredictError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def load_model(model_name: str):
    path = ARTIFACTS / model_name / "model.joblib"
    if not path.exists():
        raise PredictError(f"No artifact at {path}. Train {model_name} first.")
    return joblib.load(path)


def _probabilities(pipeline, frame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(frame)
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "named_steps"):
            classes = getattr(pipeline.named_steps["classifier"], "classes_", None)
        proba = np.asarray(proba)
        if classes is None:
            return proba
        col = {int(cls): i for i, cls in enumerate(classes)}
        ordered = np.zeros((len(frame), len(CLASSES)))
        for j, cls in enumerate(CLASSES):
            if cls in col:
                ordered[:, j] = proba[:, col[cls]]
        return ordered
    preds = pipeline.predict(frame)
    ordered = np.zeros((len(frame), len(CLASSES)))
    for i, pred in enumerate(preds):
        ordered[i, CLASSES.index(int(pred))] = 1.0
    return ordered


def _format_row(model_name: str, pred: int, proba_row: np.ndarray) -> dict:
    card_path = ARTIFACTS / model_name / "model_card.json"
    version = "1.0.0"
    if card_path.exists():
        import json

        version = json.loads(card_path.read_text(encoding="utf-8")).get("version", version)
    return {
        "model": model_name,
        "model_version": version,
        "predicted_severity": int(pred),
        "probabilities": {str(cls): float(proba_row[i]) for i, cls in enumerate(CLASSES)},
    }


def _predict_model(model_name: str, payload) -> dict:
    schema = load_schema()
    frame = validate_payload(payload, schema)
    pipeline = load_model(model_name)
    preds = np.asarray(pipeline.predict(frame)).astype(int)
    proba = _probabilities(pipeline, frame)
    results = [_format_row(model_name, pred, proba[i]) for i, pred in enumerate(preds)]
    if isinstance(payload, list) or (isinstance(payload, dict) and "instances" in payload):
        if len(results) > 1:
            return {"predictions": results}
    return results[0]


def predict_logistic_regression(payload) -> dict:
    return _predict_model("logistic_regression", payload)


def predict_svm(payload) -> dict:
    return _predict_model("svm", payload)


def predict_random_forest(payload) -> dict:
    return _predict_model("random_forest", payload)


def predict_xgboost(payload) -> dict:
    return _predict_model("xgboost", payload)


PREDICTORS = {
    "logistic_regression": predict_logistic_regression,
    "svm": predict_svm,
    "random_forest": predict_random_forest,
    "xgboost": predict_xgboost,
}


def predict(model_name: str, payload) -> dict:
    if model_name not in PREDICTORS:
        raise PredictError(f"Unknown model `{model_name}`. Choose from {sorted(PREDICTORS)}")
    return PREDICTORS[model_name](payload)


__all__ = [
    "predict",
    "predict_logistic_regression",
    "predict_svm",
    "predict_random_forest",
    "predict_xgboost",
    "load_model",
    "SchemaError",
    "PredictError",
]
