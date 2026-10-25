"""One table and chart across all trained models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.paths import KNOWN_MODELS, project_root


def compare_models(model_names: tuple[str, ...] = KNOWN_MODELS) -> pd.DataFrame:
    rows = []
    for name in model_names:
        path = project_root() / "results" / name / "metrics.json"
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    if not rows:
        raise FileNotFoundError("No results/*/metrics.json files found. Run evaluate first.")

    frame = pd.DataFrame(rows)
    out_dir = project_root() / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    frame.to_csv(csv_path, index=False)

    ax = frame.set_index("model")[["accuracy", "balanced_accuracy", "macro_f1"]].plot(
        kind="bar", figsize=(8, 4.5), rot=20
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Frozen-test comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = out_dir / "comparison.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(frame.to_string(index=False))
    print(f"wrote {csv_path} and {fig_path}")
    return frame


if __name__ == "__main__":
    compare_models()
