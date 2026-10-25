"""Fit a model on the frozen train split and write serving artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.data import load_processed_frame, load_train_test
from model.paths import KNOWN_MODELS, load_config, load_model_config, project_root
from model.registry import build_pipeline


def _strip_step_prefix(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        cleaned[key.split("__")[-1] if "__" in str(key) else key] = value
    return cleaned


def load_best_params(model_name: str) -> dict | None:
    path = project_root() / "results" / model_name / "best_params.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded.get("best_params", loaded)


def write_input_schema(cfg: dict, data) -> None:
    df = load_processed_frame(cfg)
    fields = []
    for name in cfg["data"]["feature_cols"]:
        series = df[name]
        fields.append(
            {
                "name": name,
                "type": "integer" if name == "month" else "number",
                "min": float(np.nanmin(series.to_numpy())),
                "max": float(np.nanmax(series.to_numpy())),
            }
        )
    schema = {
        "target": cfg["data"]["target_col"],
        "classes": [1, 2, 3, 4],
        "required": list(cfg["data"]["feature_cols"]),
        "forbidden": list(cfg["data"]["forbidden_cols"]),
        "fields": fields,
    }
    path = project_root() / cfg["paths"]["input_schema"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def train_model(model_name: str) -> dict:
    cfg = load_config()
    model_cfg = load_model_config(model_name)
    data = load_train_test(cfg)
    pipeline = build_pipeline(model_name, cfg, model_cfg)

    best_params = load_best_params(model_name)
    if best_params:
        pipeline.set_params(**best_params)
        print(f"Using tuned params for {model_name}: {best_params}")
    else:
        print(f"No results/{model_name}/best_params.yaml; using YAML defaults.")

    print(f"Fitting {model_name} on {len(data.X_train)} rows...")
    pipeline.fit(data.X_train, data.y_train)

    y_pred = pipeline.predict(data.X_train)
    train_metrics = {
        "accuracy": float(accuracy_score(data.y_train, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(data.y_train, y_pred)),
        "macro_f1": float(f1_score(data.y_train, y_pred, average="macro")),
    }
    print("train:", train_metrics)

    artifact_dir = project_root() / cfg["paths"]["artifacts_dir"] / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    outputs_dir = project_root() / cfg["paths"]["model_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, outputs_dir / f"{model_name}.pkl")

    write_input_schema(cfg, data)

    card = {
        "model_name": model_name,
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": data.feature_cols,
        "excluded": cfg["data"]["forbidden_cols"],
        "n_train": int(len(data.X_train)),
        "n_test": int(len(data.X_test)),
        "train_metrics": train_metrics,
        "params": _strip_step_prefix(best_params or (model_cfg.get("defaults") or {})),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "artifact": str(model_path.relative_to(project_root())).replace("\\", "/"),
    }
    (artifact_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"saved {model_path}")
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=KNOWN_MODELS)
    args = parser.parse_args()
    train_model(args.model)


if __name__ == "__main__":
    main()
