"""Hyperparameter search on frozen training locations (GroupKFold)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.data import load_train_test
from model.paths import KNOWN_MODELS, load_config, load_model_config, project_root
from model.registry import build_pipeline


def _rewrite_nulls(obj):
    if isinstance(obj, dict):
        return {k: _rewrite_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_nulls(v) for v in obj]
    return obj


def run_tuning(model_name: str) -> dict:
    cfg = load_config()
    model_cfg = load_model_config(model_name)
    data = load_train_test(cfg)
    pipeline = build_pipeline(model_name, cfg, model_cfg)

    tuning = model_cfg.get("tuning") or {}
    param_distributions = _rewrite_nulls(tuning.get("param_distributions") or {})
    cv_splits = list(
        GroupKFold(n_splits=cfg["split"]["cv_folds"]).split(
            data.X_train, data.y_train, data.groups_train
        )
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=int(tuning.get("n_iter", 8)),
        scoring=tuning.get("scoring", "f1_macro"),
        cv=cv_splits,
        random_state=cfg["split"]["random_state"],
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    print(f"Tuning {model_name} on {len(data.X_train)} train rows...")
    search.fit(data.X_train, data.y_train)

    out_dir = project_root() / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"best_params": search.best_params_, "best_cv_score": float(search.best_score_)}
    with (out_dir / "best_params.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best, handle, sort_keys=False)

    reports_dir = project_root() / cfg["paths"]["reports_dir"] / model_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "tuning_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_params": search.best_params_,
                "best_cv_score": float(search.best_score_),
            },
            handle,
            indent=2,
            default=str,
        )

    print(f"best CV {tuning.get('scoring')}: {search.best_score_:.4f}")
    print(f"best params written to {out_dir / 'best_params.yaml'}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=KNOWN_MODELS)
    args = parser.parse_args()
    run_tuning(args.model)


if __name__ == "__main__":
    main()
