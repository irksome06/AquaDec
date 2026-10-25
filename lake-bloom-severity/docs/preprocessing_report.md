# Preprocessing report

Processed table: `data/processed/clean_data.csv` (copied from the AquaDec clean dataset).

The raw join `data/raw/caml_weather_sat.csv` is not stored in this folder. When it is present, `python preprocessing/preprocess.py` rebuilds the processed file with the same rules:

1. Keep rows with a good satellite match (`status == ok`).
2. Keep rows with complete weather (`tmax_c`, `tmin_c`, `prcp_mm`, `rain_sum_7d_mm`, `tmax_mean_7d_c`).
3. Merge severity 5 into 4 → target `severity4` in `{1, 2, 3, 4}`.
4. Derive `year` and `month` from `date` (`year` is stored but **not** used as a model feature).
5. Build `location_group` from rounded lat/lon for location-level splits.
6. Drop identifiers and leakage columns: `uid`, `lat`, `lon`, `time`, `date`, `abun`, `severity`, `station_km`, `s2_item`, `status`.
7. Cap `wind_ms` at 15 m/s and fill remaining wind gaps with the median.

## Training exclusions

Models and the serving schema **do not** use:

- `year`
- `distance_to_water_m`
- `region`
- `data_provider`

Inputs at prediction time are weather, Sentinel-2 bands/indices, image-quality fields, and `month`.
