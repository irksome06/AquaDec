"""
Cleaning and preprocessing for the lake bloom project.

Usage:
    python preprocess.py caml_weather_sat.csv          (give the satellite output file)
Output:
    clean_data.csv

What it does
  1. Keeps only rows with a good satellite match (status == ok).
  2. Keeps only rows with complete weather (temperature, rainfall, 7 day rain, 7 day temperature).
  3. Merges severity 5 into 4  ->  target column `severity4` (levels 1, 2, 3, 4).
  4. Builds year and month from the date.
  5. Makes `location_group`: one id per lake location, used to split train and test by location.
  6. Drops columns that are not features or would leak the answer:
     uid, lat, lon, time, date, abun, severity, station_km, s2_item, status.
  7. Caps unrealistic wind speeds and fills the few missing wind values with the median.
"""
import sys
import numpy as np
import pandas as pd

INPUT = sys.argv[1] if len(sys.argv) > 1 else "caml_weather_sat.csv"
OUTPUT = "clean_data.csv"

# Set True to also keep only clean images (cloud under 20 %, image within 5 days of the sample).
# This gives fewer rows than 3,688.
QUALITY_FILTER = False

WEATHER_REQUIRED = ["tmax_c", "tmin_c", "prcp_mm", "rain_sum_7d_mm", "tmax_mean_7d_c"]
WIND_CAP = 15.0   # m/s, higher values were unrealistic

df = pd.read_csv(INPUT)
print("start:", df.shape)

# 1. good satellite rows
df = df[df["status"] == "ok"]
print("after satellite status == ok:", len(df))

# 2. complete weather
df = df.dropna(subset=WEATHER_REQUIRED)
print("after complete weather:", len(df))

# optional image quality filter
if QUALITY_FILTER:
    df = df[(df["s2_cloud_pct"] <= 20) & (df["s2_days_offset"].abs() <= 5)]
    print("after quality filter:", len(df))

df = df.drop_duplicates(subset="uid").copy()

# 3. severity 5 merged into 4
df["severity4"] = df["severity"].clip(upper=4).astype(int)

# 4. date features
d = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
df["year"] = d.dt.year
df["month"] = d.dt.month

# 5. location group for grouped train/test splits
loc = df["lat"].round(3).astype(str) + "_" + df["lon"].round(3).astype(str)
df["location_group"] = loc.astype("category").cat.codes

# 6. drop columns
drop_cols = ["uid", "lat", "lon", "time", "date", "abun", "severity", "station_km", "s2_item", "status"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 7. wind: cap and fill
df["wind_ms"] = df["wind_ms"].clip(upper=WIND_CAP)
n_wind = int(df["wind_ms"].isna().sum())
df["wind_ms"] = df["wind_ms"].fillna(df["wind_ms"].median())
if "distance_to_water_m" in df and df["distance_to_water_m"].isna().any():
    df["distance_to_water_m"] = df["distance_to_water_m"].fillna(df["distance_to_water_m"].median())

# column order: target and group first
first = ["severity4", "location_group"]
df = df[first + [c for c in df.columns if c not in first]]
df.to_csv(OUTPUT, index=False)

print("\nfinal shape:", df.shape, "-> saved to", OUTPUT)
print("wind values filled with median:", n_wind)
print("missing values left:", int(df.isna().sum().sum()))
print("lake locations:", df["location_group"].nunique())
print("\nclass counts (severity4):")
print(df["severity4"].value_counts().sort_index().to_string())
print("\ncolumns:", df.columns.tolist())
