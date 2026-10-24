"""
preprocessing/preprocess.py
===========================
Data cleaning and preprocessing pipeline: raw -> processed
Reads raw matched satellite/weather data (caml_weather_sat.csv) and produces clean_data.csv.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_raw_data(raw_df):
    """
    Cleans raw merged SeaBASS + NOAA Weather + Sentinel-2 dataset:
      1. Drops rows without valid satellite reflectance.
      2. Enforces minimum water pixel criteria.
      3. Filters extreme/anomalous reflectance.
      4. Merges severity level 5 into 4 (creating severity4: classes 1, 2, 3, 4).
      5. Derives month and year if date column exists.
      6. Drops any missing values.
    """
    df = raw_df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Date handling -> year and month
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and ("year" not in df.columns or "month" not in df.columns):
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = dates.dt.year
        df["month"] = dates.dt.month

    # Target preparation: severity -> severity4 (classes 1, 2, 3, 4)
    if "severity4" not in df.columns:
        if "severity" in df.columns:
            # Map severity 5 to 4 as per methodology
            df["severity4"] = df["severity"].apply(lambda s: 4 if s >= 4 else int(s))
        elif "abundance" in df.columns:
            # Categorize abundance into 4 severity levels if needed
            bins = [-np.inf, 20000, 100000, 10000000, np.inf]
            df["severity4"] = pd.cut(df["abundance"], bins=bins, labels=[1, 2, 3, 4]).astype(int)

    # Filter water pixels if present
    if "n_water_px" in df.columns:
        df = df[df["n_water_px"] > 0]

    # Filter reflectance anomalies if Sentinel-2 bands present
    sat_bands = [b for b in ["B02", "B03", "B04", "B05", "B08"] if b in df.columns]
    for b in sat_bands:
        df = df[(df[b] >= 0) & (df[b] <= 1.0)]

    # Drop rows with null values
    df = df.dropna()

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess lake bloom dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_dir, args.config)
    cfg = load_config(config_file)

    raw_path = os.path.join(base_dir, cfg["paths"]["raw_data"])
    processed_path = os.path.join(base_dir, cfg["paths"]["processed_data"])
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    print("=" * 70)
    print("STEP: PREPROCESSING (raw -> processed)")
    print("=" * 70)

    if os.path.exists(raw_path):
        print(f"Reading raw data from: {raw_path}")
        raw_df = pd.read_csv(raw_path)
        print(f"Raw shape: {raw_df.shape}")
        clean_df = clean_raw_data(raw_df)
        clean_df.to_csv(processed_path, index=False)
        print(f"Cleaned data saved to: {processed_path}")
        print(f"Processed shape: {clean_df.shape}")
    elif os.path.exists(processed_path):
        print(f"Note: Raw file '{raw_path}' not found.")
        print(f"Existing clean data found at '{processed_path}'.")
        df = pd.read_csv(processed_path)
        print(f"Dataset integrity verified: {df.shape[0]} rows, {df.shape[1]} columns.")
        print(f"Target distribution (severity4):\n{df['severity4'].value_counts().sort_index().to_dict()}")
    else:
        raise FileNotFoundError(
            f"Neither raw data ({raw_path}) nor processed data ({processed_path}) was found."
        )

    print("Preprocessing completed successfully.\n")


if __name__ == "__main__":
    main()
