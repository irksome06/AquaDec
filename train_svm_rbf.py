"""Train the lake bloom severity SVM (RBF kernel) and save it as a .pkl file.

Usage:
    python train_svm_rbf.py clean_data.csv
    python train_svm_rbf.py clean_data.csv --output svm_rbf_final.pkl
"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42

# Full feature set (metadata + weather + satellite + image-quality)
META = ["region", "data_provider", "distance_to_water_m", "year", "month"]
WEATHER = ["tmax_c", "tmin_c", "prcp_mm", "wind_ms", "rain_sum_7d_mm", "tmax_mean_7d_c"]
SPECTRAL = ["B02", "B03", "B04", "B05", "B08", "NDCI", "NDVI", "NDWI"]
QUALITY = ["n_water_px", "s2_days_offset", "s2_cloud_pct"]
FEATURES = META + WEATHER + SPECTRAL + QUALITY

LOG_COLS = ["prcp_mm", "rain_sum_7d_mm", "distance_to_water_m", "n_water_px"]
CAT_COLS = ["region", "data_provider"]
CYC_COLS = ["month"]

PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
}


def month_to_cyclic(a):
    """Map month to (sin, cos) so December and January are close."""
    a = np.asarray(a, dtype=float)
    return np.hstack([np.sin(2 * np.pi * a / 12), np.cos(2 * np.pi * a / 12)])


def build_pipeline(cols):
    log_c = [c for c in LOG_COLS if c in cols]
    cat_c = [c for c in CAT_COLS if c in cols]
    cyc_c = [c for c in CYC_COLS if c in cols]
    num_c = [c for c in cols if c not in log_c + cat_c + cyc_c]

    parts = []
    if log_c:
        parts.append(("log", Pipeline([("log1p", FunctionTransformer(np.log1p)),
                                       ("scale", StandardScaler())]), log_c))
    if num_c:
        parts.append(("num", StandardScaler(), num_c))
    if cat_c:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_c))
    if cyc_c:
        parts.append(("month", FunctionTransformer(month_to_cyclic), cyc_c))

    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("svm", SVC(kernel="rbf", class_weight="balanced"))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to clean_data.csv")
    parser.add_argument("--output", default="svm_rbf_final.pkl", help="Output .pkl path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    y = df["severity4"]
    groups = df["location_group"]  # used only for splitting, never as a feature
    X = df[FEATURES]

    search = GridSearchCV(
        build_pipeline(FEATURES), PARAM_GRID,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1_macro", n_jobs=-1,
    ).fit(X, y, groups=groups)

    print("Best params:", search.best_params_, f"(CV macro F1 = {search.best_score_:.3f})")

    with open(args.output, "wb") as f:
        pickle.dump(search.best_estimator_, f)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    # month_to_cyclic lives in this module, so load the .pkl with this file importable
    # (e.g. `from train_svm_rbf import month_to_cyclic` before pickle.load).
    main()
