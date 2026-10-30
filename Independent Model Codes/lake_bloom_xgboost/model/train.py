"""
model/train.py
==============
Final model training stage for XGBoost.
Loads hyperparameters from configs/best_params.yaml (or base params if not yet tuned),
trains the model on the full training set with class-balanced weighting,
and serializes the artifact to outputs/models/xgb_model.pkl.
"""

import os
import sys
import yaml
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
        compute_balanced_weights,
        save_artifact,
        get_base_dir,
    )
except ImportError:
    from utils import (
        load_config,
        load_dataset,
        get_train_test_split,
        build_preprocessor,
        compute_balanced_weights,
        save_artifact,
        get_base_dir,
    )


def train_model():
    print("=" * 70)
    print("STAGE: MODEL TRAINING (outputs/models/xgb_model.pkl)")
    print("=" * 70)

    cfg = load_config()
    base_dir = get_base_dir()

    # Load parameters
    best_params_path = os.path.join(base_dir, cfg["paths"]["best_params_path"])
    if os.path.exists(best_params_path):
        print(f"Loading tuned hyperparameters from: {best_params_path}")
        with open(best_params_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            model_params = loaded.get("best_params", loaded)
    else:
        print("Tuned parameters file not found; falling back to base parameters in config.yaml.")
        model_params = cfg["model"]["base_params"]

    print("Active XGBoost Parameters:")
    for k, v in model_params.items():
        print(f"  • {k}: {v}")

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

    xgb_clf = XGBClassifier(
        objective=cfg["model"]["objective"],
        eval_metric=cfg["model"]["eval_metric"],
        num_class=cfg["model"]["num_class"],
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        **model_params,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_clf),
    ])

    print("\nFitting pipeline on training data...")
    sample_weights = compute_balanced_weights(y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    # Train evaluations
    y_pred_train = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    train_macro_f1 = f1_score(y_train, y_pred_train, average="macro")

    print(f"\nTraining Results:")
    print(f"  • Train Accuracy         : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  • Train Balanced Accuracy: {train_bal_acc:.4f}")
    print(f"  • Train Macro F1         : {train_macro_f1:.4f}")

    # Bundle pipeline with metadata for robust evaluation
    artifact = {
        "pipeline": pipeline,
        "label_map": label_map,
        "inv_label_map": inv_label_map,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "model_params": model_params,
    }

    model_rel_path = cfg["paths"]["model_path"]
    save_artifact(artifact, model_rel_path)
    print("Model training stage complete.\n")

    return pipeline


if __name__ == "__main__":
    train_model()
