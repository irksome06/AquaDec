"""
Cleaning and preprocessing for lake bloom severity.

Usage (from lake-bloom-severity/):
    python preprocessing/preprocess.py
    python preprocessing/preprocess.py path/to/caml_weather_sat.csv

Reads data/raw/caml_weather_sat.csv by default and writes
data/processed/clean_data.csv plus docs/preprocessing_report.md.

If the raw file is missing, the existing processed file is left in place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.paths import load_config, project_root

QUALITY_FILTER = False
WEATHER_REQUIRED = ["tmax_c", "tmin_c", "prcp_mm", "rain_sum_7d_mm", "tmax_mean_7d_c"]
WIND_CAP = 15.0


def preprocess(raw_path: Path, output_path: Path) -> pd.DataFrame:
    lines: list[str] = []
    df = pd.read_csv(raw_path)
    lines.append(f"start: {df.shape}")

    df = df[df["status"] == "ok"]
    lines.append(f"after satellite status == ok: {len(df)}")

    df = df.dropna(subset=WEATHER_REQUIRED)
    lines.append(f"after complete weather: {len(df)}")

    if QUALITY_FILTER:
        df = df[(df["s2_cloud_pct"] <= 20) & (df["s2_days_offset"].abs() <= 5)]
        lines.append(f"after quality filter: {len(df)}")

    df = df.drop_duplicates(subset="uid").copy()

    df["severity4"] = df["severity"].clip(upper=4).astype(int)

    dates = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month

    loc = df["lat"].round(3).astype(str) + "_" + df["lon"].round(3).astype(str)
    df["location_group"] = loc.astype("category").cat.codes

    drop_cols = [
        "uid",
        "lat",
        "lon",
        "time",
        "date",
        "abun",
        "severity",
        "station_km",
        "s2_item",
        "status",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["wind_ms"] = df["wind_ms"].clip(upper=WIND_CAP)
    n_wind = int(df["wind_ms"].isna().sum())
    df["wind_ms"] = df["wind_ms"].fillna(df["wind_ms"].median())
    if "distance_to_water_m" in df.columns and df["distance_to_water_m"].isna().any():
        df["distance_to_water_m"] = df["distance_to_water_m"].fillna(
            df["distance_to_water_m"].median()
        )

    first = ["severity4", "location_group"]
    df = df[first + [c for c in df.columns if c not in first]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    counts = df["severity4"].value_counts().sort_index()
    report = [
        "# Preprocessing report",
        "",
        f"Input: `{raw_path}`",
        f"Output: `{output_path}`",
        "",
        *[f"- {line}" for line in lines],
        f"- wind values filled with median: {n_wind}",
        f"- missing values left: {int(df.isna().sum().sum())}",
        f"- lake locations: {df['location_group'].nunique()}",
        f"- final shape: {df.shape}",
        "",
        "## Class counts (`severity4`)",
        "",
        counts.to_string(),
        "",
        "## Columns",
        "",
        ", ".join(df.columns.tolist()),
        "",
        "Training does **not** use `year`, `distance_to_water_m`, `region`, or `data_provider`.",
        "",
    ]
    report_path = project_root() / "docs" / "preprocessing_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(lines))
    print(f"final shape: {df.shape} -> saved to {output_path}")
    print(f"wrote {report_path}")
    return df


def main() -> None:
    cfg = load_config()
    root = project_root()
    parser = argparse.ArgumentParser(description="Raw satellite/weather CSV to clean_data.csv")
    parser.add_argument(
        "raw",
        nargs="?",
        default=str(root / cfg["paths"]["raw_data"]),
        help="Path to caml_weather_sat.csv",
    )
    args = parser.parse_args()
    raw_path = Path(args.raw)
    output_path = root / cfg["paths"]["processed_data"]

    if not raw_path.exists():
        if output_path.exists():
            print(f"Raw file not found at {raw_path}. Keeping existing {output_path}.")
            return
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}. Place caml_weather_sat.csv under data/raw/."
        )

    preprocess(raw_path, output_path)


if __name__ == "__main__":
    main()
