"""
model/finetune.py
=================
Hyperparameter search for XGBoost on Lake Bloom severity prediction.
Uses 5-fold GroupKFold cross-validation on the training data.
Saves optimal parameters to configs/best_params.yaml and reports to outputs/reports/.
"""

import os
import sys
import json
import yaml
import warnings
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Local package imports
try:
    from model.utils import (
        load_config,
        load_dataset,
        get_train_test_split,
        build_preprocessor,
        get_base_dir,
    )
except ImportError:
    from utils import (
        load_config,
        load_dataset,
        get_train_test_split,
        build_preprocessor,
        get_base_dir,
    )


def run_tuning():
    print("=" * 70)
    print("STAGE: HYPERPARAMETER TUNING (GroupKFold Cross-Validation)")
    print("=" * 70)

    cfg = load_config()
    base_dir = get_base_dir()

    df = load_dataset(cfg)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_cols,
        label_map,
        inv_label_map,
    ) = get_train_test_split(df, cfg)

    cat_cols = X_train.select_dtypes(include=["object", "category", "string", str]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = build_preprocessor(cat_cols, num_cols)

    base_xgb = XGBClassifier(
        objective=cfg["model"]["objective"],
        eval_metric=cfg["model"]["eval_metric"],
        num_class=cfg["model"]["num_class"],
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_xgb),
    ])

    param_distributions = cfg["tuning"]["param_distributions"]
    n_iter = cfg["tuning"].get("n_iter", 12)
    cv_folds = cfg["split"].get("cv_folds", 5)

    gkf = GroupKFold(n_splits=cv_folds)

    print(f"Running RandomizedSearchCV ({n_iter} iterations, {cv_folds}-fold GroupKFold)...")
    print(f"Scoring metric: {cfg['tuning']['scoring']}")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=gkf,
        scoring=cfg["tuning"]["scoring"],
        random_state=cfg["tuning"]["random_state"],
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train, groups=groups_train)

    best_cv_score = float(search.best_score_)
    print(f"\nBest Cross-Validation Macro F1 Score: {best_cv_score:.4f}")
    print(f"Best Parameter Combination:")
    clean_best_params = {}
    for k, v in search.best_params_.items():
        clean_key = k.replace("classifier__", "")
        clean_best_params[clean_key] = int(v) if isinstance(v, (int, complex)) else float(v) if isinstance(v, float) else v
        print(f"  • {clean_key}: {v}")

    # Save to configs/best_params.yaml
    best_params_output = {
        "best_params": clean_best_params,
        "best_cv_macro_f1": best_cv_score,
        "cv_folds": cv_folds,
        "n_iter_searched": n_iter,
    }

    best_params_file = os.path.join(base_dir, cfg["paths"]["best_params_path"])
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    with open(best_params_file, "w", encoding="utf-8") as f:
        yaml.dump(best_params_output, f, default_flow_style=False)
    print(f"\nSaved optimal parameters to: {best_params_file}")

    # Save tuning summary report
    reports_dir = os.path.join(base_dir, cfg["paths"]["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)
    tuning_report_file = os.path.join(reports_dir, "tuning_results.json")
    with open(tuning_report_file, "w", encoding="utf-8") as f:
        json.dump(best_params_output, f, indent=2)
    print(f"Saved tuning log to: {tuning_report_file}")
    print("Hyperparameter tuning stage complete.\n")

    return clean_best_params


if __name__ == "__main__":
    run_tuning()
