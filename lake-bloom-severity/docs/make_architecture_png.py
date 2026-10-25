"""Write docs/architecture.png (no extra graph libraries)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")

BOXES = [
    (0.05, 0.62, "raw CSV\ncaml_weather_sat.csv"),
    (0.28, 0.62, "preprocess.py\nclean_data.csv"),
    (0.51, 0.62, "make_split.py\nsplits.csv"),
    (0.74, 0.62, "model/data.py\nchecksum + frozen split"),
    (0.05, 0.28, "registry.py\nLR / SVM / RF / XGB"),
    (0.28, 0.28, "finetune + train\nartifacts/*.joblib"),
    (0.51, 0.28, "evaluate + compare\nresults/"),
    (0.74, 0.28, "serving/predictor.py\nweather + satellite JSON"),
]


def box(ax, x, y, text):
    patch = FancyBboxPatch(
        (x, y),
        0.21,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1f4e79",
        facecolor="#e8f1fa",
    )
    ax.add_patch(patch)
    ax.text(x + 0.105, y + 0.11, text, ha="center", va="center", fontsize=8)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Lake bloom severity: train once, serve weather + satellite JSON")
    for x, y, text in BOXES:
        box(ax, x, y, text)
    arrows = [
        ((0.26, 0.73), (0.28, 0.73)),
        ((0.49, 0.73), (0.51, 0.73)),
        ((0.72, 0.73), (0.74, 0.73)),
        ((0.845, 0.62), (0.845, 0.50)),
        ((0.845, 0.50), (0.155, 0.50)),
        ((0.155, 0.50), (0.155, 0.50)),
        ((0.155, 0.50), (0.155, 0.50)),
    ]
    ax.annotate("", xy=(0.28, 0.73), xytext=(0.26, 0.73), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.51, 0.73), xytext=(0.49, 0.73), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.74, 0.73), xytext=(0.72, 0.73), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.155, 0.50), xytext=(0.845, 0.62), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.28, 0.39), xytext=(0.26, 0.39), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.51, 0.39), xytext=(0.49, 0.39), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.74, 0.39), xytext=(0.72, 0.39), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
