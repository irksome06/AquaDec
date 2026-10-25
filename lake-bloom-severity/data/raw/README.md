# Raw data

Place `caml_weather_sat.csv` here (NASA SeaBASS / NOAA GHCN / Sentinel-2 join).

```bash
python preprocessing/preprocess.py
```

If this file is absent, the committed `data/processed/clean_data.csv` is used as-is.
