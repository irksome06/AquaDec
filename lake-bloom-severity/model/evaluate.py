"""Frozen-test metrics, dummy baseline, optional repeated train splits, plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.data import load_train_test
from model.paths import (
    KNOWN_MODELS,
    load_config,
    load_evaluation_config,
    load_model_config,
    project_root,
)
from model.registry import build_pipeline
from model.train import load_best_params


def _metric_block(y_true, y_pred) -> dict:
    labels = [1, 2, 3, 4]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": per_class,
    }


def _save_confusion(y_true, y_pred, path: Path, title: str) -> None:
    labels = [1, 2, 3, 4]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted severity")
    plt.ylabel("True severity")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def evaluate_model(model_name: str) -> dict:
    cfg = load_config()
    eval_cfg = load_evaluation_config()
    data = load_train_test(cfg)
    artifact = project_root() / cfg["paths"]["artifacts_dir"] / model_name / "model.joblib"
    pipeline = joblib.load(artifact)

    y_pred = pipeline.predict(data.X_test)
    metrics = _metric_block(data.y_test, y_pred)
    metrics["n_test"] = int(len(data.y_test))
    metrics["classification_report"] = classification_report(
        data.y_test, y_pred, labels=[1, 2, 3, 4], output_dict=True, zero_division=0
    )

    if eval_cfg.get("comparisons", {}).get("dummy_baseline", {}).get("enabled", True):
        dummy = DummyClassifier(
            strategy=eval_cfg["comparisons"]["dummy_baseline"].get("strategy", "most_frequent"),
            random_state=cfg["split"]["random_state"],
        )
        dummy.fit(data.X_train, data.y_train)
        metrics["dummy_baseline"] = _metric_block(data.y_test, dummy.predict(data.X_test))

    repeated = eval_cfg.get("repeated_splits") or {}
    if repeated.get("enabled"):
        model_cfg = load_model_config(model_name)
        best_params = load_best_params(model_name)
        splitter = GroupShuffleSplit(
            n_splits=int(repeated.get("n_splits", 3)),
            test_size=repeated.get("test_size", 0.2),
            random_state=repeated.get("random_state", 42),
        )
        scores = []
        for train_idx, valid_idx in splitter.split(data.X_train, data.y_train, data.groups_train):
            pipe = build_pipeline(model_name, cfg, model_cfg)
            if best_params:
                pipe.set_params(**best_params)
            pipe.fit(data.X_train.iloc[train_idx], data.y_train.iloc[train_idx])
            pred = pipe.predict(data.X_train.iloc[valid_idx])
            scores.append(f1_score(data.y_train.iloc[valid_idx], pred, average="macro"))
        metrics["repeated_train_split_macro_f1"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": [float(s) for s in scores],
        }

    out_dir = project_root() / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_name = eval_cfg.get("plots", {}).get("confusion_matrix", "confusion_matrix.png")
    _save_confusion(
        data.y_test,
        y_pred,
        out_dir / plot_name,
        f"{model_name} confusion matrix (frozen test)",
    )
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=KNOWN_MODELS)
    args = parser.parse_args()
    evaluate_model(args.model)


if __name__ == "__main__":
    main()
