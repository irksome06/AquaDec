"""
model/evaluate.py
=================
Evaluation stage for XGBoost Lake Bloom severity classification.
Generates test metrics, dummy baseline comparison, feature ablation,
and publication-grade diagnostic plots saved to outputs/reports/.
"""

import os
import sys
import json
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.dummy import DummyClassifier
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
        load_artifact,
        build_preprocessor,
        compute_balanced_weights,
        get_base_dir,
    )
except ImportError:
    from utils import (
        load_config,
        load_dataset,
        get_train_test_split,
        load_artifact,
        build_preprocessor,
        compute_balanced_weights,
        get_base_dir,
    )


def evaluate_model():
    print("=" * 70)
    print("STAGE: MODEL EVALUATION & REPORTING (outputs/reports/)")
    print("=" * 70)

    cfg = load_config()
    eval_cfg = load_config("configs/evaluation.yaml")
    base_dir = get_base_dir()

    reports_dir = os.path.join(base_dir, cfg["paths"]["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)

    # Load model artifact
    model_artifact = load_artifact(cfg["paths"]["model_path"])
    pipeline = model_artifact["pipeline"]
    label_map = model_artifact["label_map"]
    inv_label_map = model_artifact["inv_label_map"]

    df = load_dataset(cfg)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_cols,
        _,
        _,
    ) = get_train_test_split(df, cfg)

    # Predictions
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    # Convert back to original 1..4 labels for reporting
    y_test_orig = y_test.map(inv_label_map)
    y_pred_test_orig = pd.Series(y_pred_test).map(inv_label_map)
    y_train_orig = y_train.map(inv_label_map)
    y_pred_train_orig = pd.Series(y_pred_train).map(inv_label_map)

    # Test Metrics
    test_acc = accuracy_score(y_test_orig, y_pred_test_orig)
    test_bal_acc = balanced_accuracy_score(y_test_orig, y_pred_test_orig)
    test_macro_f1 = f1_score(y_test_orig, y_pred_test_orig, average="macro")
    test_weighted_f1 = f1_score(y_test_orig, y_pred_test_orig, average="weighted")

    train_acc = accuracy_score(y_train_orig, y_pred_train_orig)
    train_bal_acc = balanced_accuracy_score(y_train_orig, y_pred_train_orig)
    train_macro_f1 = f1_score(y_train_orig, y_pred_train_orig, average="macro")

    classes_orig = [1, 2, 3, 4]
    prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_test_orig, y_pred_test_orig, labels=classes_orig
    )

    cm = confusion_matrix(y_test_orig, y_pred_test_orig, labels=classes_orig)

    print(f"\nFinal Test Set Performance (XGBoost):")
    print(f"  • Accuracy         : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  • Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"  • Macro F1 Score   : {test_macro_f1:.4f}")
    print(f"  • Weighted F1 Score: {test_weighted_f1:.4f}")
    print(f"\nClassification Report:\n", classification_report(y_test_orig, y_pred_test_orig, digits=4))

    # Baseline Comparison: Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train_orig)
    y_dummy_pred = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test_orig, y_dummy_pred)
    dummy_macro_f1 = f1_score(y_test_orig, y_dummy_pred, average="macro")
    dummy_bal_acc = balanced_accuracy_score(y_test_orig, y_dummy_pred)

    print(f"Dummy Classifier (most_frequent):")
    print(f"  • Accuracy: {dummy_acc:.4f} | Balanced Acc: {dummy_bal_acc:.4f} | Macro F1: {dummy_macro_f1:.4f}")

    # Feature Ablation: Without data_provider
    print("\nRunning Feature Ablation (re-training WITHOUT 'data_provider')...")
    (
        X_train_no_dp,
        X_test_no_dp,
        y_train_no_dp,
        y_test_no_dp,
        _,
        _,
        feat_cols_no_dp,
        _,
        _,
    ) = get_train_test_split(df, cfg, exclude_feature="data_provider")

    cat_no_dp = X_train_no_dp.select_dtypes(include=["object", "category", "string", str]).columns.tolist()
    num_no_dp = [c for c in X_train_no_dp.columns if c not in cat_no_dp]
    prep_no_dp = build_preprocessor(cat_no_dp, num_no_dp)

    xgb_no_dp = XGBClassifier(
        objective=cfg["model"]["objective"],
        eval_metric=cfg["model"]["eval_metric"],
        num_class=cfg["model"]["num_class"],
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        **model_artifact["model_params"],
    )

    pipe_no_dp = Pipeline([("prep", prep_no_dp), ("clf", xgb_no_dp)])
    pipe_no_dp.fit(X_train_no_dp, y_train_no_dp, clf__sample_weight=compute_balanced_weights(y_train_no_dp))
    y_pred_no_dp = pd.Series(pipe_no_dp.predict(X_test_no_dp)).map(inv_label_map)

    no_dp_acc = accuracy_score(y_test_orig, y_pred_no_dp)
    no_dp_bal_acc = balanced_accuracy_score(y_test_orig, y_pred_no_dp)
    no_dp_macro_f1 = f1_score(y_test_orig, y_pred_no_dp, average="macro")
    no_dp_weighted_f1 = f1_score(y_test_orig, y_pred_no_dp, average="weighted")

    print(f"Without data_provider -> Acc: {no_dp_acc:.4f} | Bal Acc: {no_dp_bal_acc:.4f} | Macro F1: {no_dp_macro_f1:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Save Metrics & Reports
    # ──────────────────────────────────────────────────────────────────────────
    metrics_data = {
        "model": "XGBoost",
        "test_metrics": {
            "accuracy": float(test_acc),
            "balanced_accuracy": float(test_bal_acc),
            "macro_f1": float(test_macro_f1),
            "weighted_f1": float(test_weighted_f1),
        },
        "train_metrics": {
            "accuracy": float(train_acc),
            "balanced_accuracy": float(train_bal_acc),
            "macro_f1": float(train_macro_f1),
        },
        "per_class": {
            f"Class_{c}": {
                "precision": float(prec_cls[i]),
                "recall": float(rec_cls[i]),
                "f1_score": float(f1_cls[i]),
                "support": int(supp_cls[i]),
            }
            for i, c in enumerate(classes_orig)
        },
        "dummy_baseline": {
            "accuracy": float(dummy_acc),
            "balanced_accuracy": float(dummy_bal_acc),
            "macro_f1": float(dummy_macro_f1),
        },
        "ablation_without_data_provider": {
            "accuracy": float(no_dp_acc),
            "balanced_accuracy": float(no_dp_bal_acc),
            "macro_f1": float(no_dp_macro_f1),
            "weighted_f1": float(no_dp_weighted_f1),
            "delta_macro_f1": float(no_dp_macro_f1 - test_macro_f1),
            "delta_accuracy": float(no_dp_acc - test_acc),
        },
    }

    metrics_json_path = os.path.join(base_dir, eval_cfg["reports"]["metrics_json"])
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_json_path}")

    ablation_json_path = os.path.join(base_dir, eval_cfg["reports"]["ablation_json"])
    with open(ablation_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data["ablation_without_data_provider"], f, indent=2)
    print(f"Saved: {ablation_json_path}")

    # Summary Text Report
    summary_txt_path = os.path.join(base_dir, eval_cfg["reports"]["summary_text"])
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("           XGBOOST MODEL EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Accuracy         : {test_acc:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_bal_acc:.4f}\n")
        f.write(f"Test Macro F1         : {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted F1      : {test_weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test_orig, y_pred_test_orig, digits=4) + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Baseline Comparison (Dummy Classifier):\n")
        f.write(f"  Dummy Accuracy: {dummy_acc:.4f} | Macro F1: {dummy_macro_f1:.4f}\n\n")
        f.write("Ablation Study (Without data_provider):\n")
        f.write(f"  Accuracy: {no_dp_acc:.4f} (Delta: {no_dp_acc - test_acc:+.4f})\n")
        f.write(f"  Macro F1: {no_dp_macro_f1:.4f} (Delta: {no_dp_macro_f1 - test_macro_f1:+.4f})\n")
    print(f"Saved: {summary_txt_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Generate Plots
    # ──────────────────────────────────────────────────────────────────────────
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {c}" for c in classes_orig],
        yticklabels=[f"Class {c}" for c in classes_orig],
        linewidths=0.5,
    )
    plt.title("Confusion Matrix – XGBoost (severity4)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Severity", fontsize=11)
    plt.ylabel("Actual Severity", fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(base_dir, eval_cfg["plots"]["confusion_matrix"])
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved plot: {p1}")

    # Plot 2: Normalized Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    annot_m = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot_m[r, c] = f"{cm[r, c]}\n({cm_norm[r, c]:.1f}%)"

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm_norm,
        annot=annot_m,
        fmt="",
        cmap="Blues",
        xticklabels=[f"Class {c}" for c in classes_orig],
        yticklabels=[f"Class {c}" for c in classes_orig],
        cbar_kws={"label": "Row Recall (%)"},
        linewidths=0.5,
    )
    plt.title("Normalized Confusion Matrix (Recall %) – XGBoost", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Severity", fontsize=11)
    plt.ylabel("Actual Severity", fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(base_dir, eval_cfg["plots"]["normalized_confusion_matrix"])
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved plot: {p2}")

    # Plot 3: Feature Importances
    feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6.5))
    top15 = fi_df.head(15).sort_values("Importance", ascending=True)
    palette = sns.color_palette("viridis", n_colors=15)
    bars = plt.barh(top15["Feature"], top15["Importance"], color=palette, edgecolor="black", linewidth=0.5)
    for b in bars:
        w = b.get_width()
        plt.annotate(f"{w:.3f}", (w, b.get_y() + b.get_height() / 2), (4, 0), textcoords="offset points", va="center", fontsize=8.5, fontweight="bold")
    plt.xlim(0, max(top15["Importance"]) * 1.15)
    plt.title("Top 15 Feature Importances – XGBoost", fontsize=13, fontweight="bold")
    plt.xlabel("Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = os.path.join(base_dir, eval_cfg["plots"]["feature_importance"])
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"Saved plot: {p3}")

    # Plot 4: Per-Class Metrics Bar Chart
    x_pos = np.arange(len(classes_orig))
    width = 0.25
    plt.figure(figsize=(8.5, 5.5))
    plt.bar(x_pos - width, prec_cls, width=width, label="Precision", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    plt.bar(x_pos, rec_cls, width=width, label="Recall", color="#2a9d8f", edgecolor="black", linewidth=0.5)
    plt.bar(x_pos + width, f1_cls, width=width, label="F1-score", color="#e76f51", edgecolor="black", linewidth=0.5)
    plt.xticks(x_pos, [f"Class {c}" for c in classes_orig], fontsize=11, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.ylabel("Score", fontsize=11)
    plt.title("Per-Class Metrics – XGBoost on Held-Out Test Set", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4 = os.path.join(base_dir, eval_cfg["plots"]["per_class_metrics"])
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"Saved plot: {p4}")

    # Plot 5: Feature Importances Without Data Provider
    feat_names_no_dp = pipe_no_dp.named_steps["prep"].get_feature_names_out()
    imp_no_dp = pipe_no_dp.named_steps["clf"].feature_importances_
    fi_no_dp_df = pd.DataFrame({"Feature": feat_names_no_dp, "Importance": imp_no_dp}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6.5))
    top15_no_dp = fi_no_dp_df.head(15).sort_values("Importance", ascending=True)
    palette_no_dp = sns.color_palette("mako", n_colors=15)
    bars_nd = plt.barh(top15_no_dp["Feature"], top15_no_dp["Importance"], color=palette_no_dp, edgecolor="black", linewidth=0.5)
    for b in bars_nd:
        w = b.get_width()
        plt.annotate(f"{w:.3f}", (w, b.get_y() + b.get_height() / 2), (4, 0), textcoords="offset points", va="center", fontsize=8.5, fontweight="bold")
    plt.xlim(0, max(top15_no_dp["Importance"]) * 1.15)
    plt.title("Top 15 Feature Importances (XGBoost WITHOUT data_provider)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p5 = os.path.join(base_dir, eval_cfg["plots"]["feature_importance_without_data_provider"])
    plt.savefig(p5, dpi=200)
    plt.close()
    print(f"Saved plot: {p5}")

    print("Model evaluation stage completed successfully.\n")


if __name__ == "__main__":
    evaluate_model()
