"""Build sklearn pipelines for each named model."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from model.features import build_preprocessor, maybe_feature_selector
from model.paths import load_config, load_model_config


class LabelMappedClassifier(BaseEstimator, ClassifierMixin):
    """Map labels 1-4 to 0-3 for estimators that require contiguous class ids."""

    def __init__(self, estimator=None, classes=(1, 2, 3, 4), balanced_weights=False):
        self.estimator = estimator
        self.classes = classes
        self.balanced_weights = balanced_weights

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("estimator is required")
        self.estimator_ = clone(self.estimator)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array(self.classes)
        mapping = {int(cls): idx for idx, cls in enumerate(self.classes_)}
        y_mapped = np.array([mapping[int(v)] for v in y])
        fit_kwargs = {}
        if self.balanced_weights:
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_mapped)
        self.estimator_.fit(X, y_mapped, **fit_kwargs)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        mapped = np.asarray(self.estimator_.predict(X)).astype(int)
        return self.classes_[mapped]

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def _estimator(model_name: str, model_cfg: dict, random_state: int):
    defaults = dict(model_cfg.get("defaults") or {})
    defaults.setdefault("random_state", random_state)

    if model_name == "logistic_regression":
        defaults.pop("random_state", None)
        if "random_state" in (model_cfg.get("defaults") or {}):
            defaults["random_state"] = model_cfg["defaults"]["random_state"]
        return LogisticRegression(**defaults)

    if model_name == "svm":
        defaults.pop("probability", None)
        inner = SVC(**defaults)
        return CalibratedClassifierCV(inner, method="sigmoid", ensemble=False, cv=3)

    if model_name == "random_forest":
        return RandomForestClassifier(**defaults)

    if model_name == "xgboost":
        inner = XGBClassifier(**defaults)
        return LabelMappedClassifier(
            estimator=inner,
            classes=(1, 2, 3, 4),
            balanced_weights=bool(model_cfg.get("balanced_sample_weight", True)),
        )

    raise KeyError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, cfg: dict | None = None, model_cfg: dict | None = None) -> Pipeline:
    cfg = cfg or load_config()
    model_cfg = model_cfg or load_model_config(model_name)
    feature_cols = list(cfg["data"]["feature_cols"])
    prep_cfg = model_cfg.get("preprocessing") or {}
    preprocessor = build_preprocessor(feature_cols, prep_cfg)
    steps = [("preprocessor", preprocessor)]
    selector = maybe_feature_selector(prep_cfg)
    if selector is not None:
        steps.append(("select", selector))
    steps.append(
        (
            "classifier",
            _estimator(model_name, model_cfg, cfg["split"]["random_state"]),
        )
    )
    return Pipeline(steps)
