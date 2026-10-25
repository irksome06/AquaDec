"""Feature lists, log1p, cyclic month, optional SelectKBest."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LOG_COLS = ["prcp_mm", "rain_sum_7d_mm", "n_water_px"]
MONTH_COL = "month"


def month_to_cyclic(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    angle = 2.0 * np.pi * arr / 12.0
    return np.hstack([np.sin(angle), np.cos(angle)])


def _safe_log1p(values):
    arr = np.asarray(values, dtype=float)
    return np.log1p(np.clip(arr, a_min=0.0, a_max=None))


def build_preprocessor(feature_cols: list[str], prep_cfg: dict) -> ColumnTransformer:
    log_cols = [c for c in LOG_COLS if c in feature_cols] if prep_cfg.get("log_transform") else []
    use_cyclic = bool(prep_cfg.get("cyclic_month")) and MONTH_COL in feature_cols
    scale = bool(prep_cfg.get("scale"))

    skip = set(log_cols)
    if use_cyclic:
        skip.add(MONTH_COL)
    rest = [c for c in feature_cols if c not in skip]

    transformers = []
    if log_cols:
        log_steps = [("log1p", FunctionTransformer(_safe_log1p, validate=True))]
        if scale:
            log_steps.append(("scale", StandardScaler()))
        transformers.append(("log", Pipeline(log_steps), log_cols))
    if rest:
        if scale:
            transformers.append(("num", StandardScaler(), rest))
        else:
            transformers.append(("num", "passthrough", rest))
    if use_cyclic:
        transformers.append(
            (
                "month",
                FunctionTransformer(month_to_cyclic, validate=False),
                [MONTH_COL],
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def maybe_feature_selector(prep_cfg: dict):
    fs = prep_cfg.get("feature_selection") or {}
    if not fs.get("enabled"):
        return None
    return SelectKBest(score_func=f_classif, k=int(fs.get("k", 12)))
