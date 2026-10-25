# Backend integration

This folder is the only interface your service needs. There is no HTTP server here: load the joblib pipeline (or call the Python helpers) from your API process.

## Layout

- `predictor.py` — `predict_logistic_regression`, `predict_svm`, `predict_random_forest`, `predict_xgboost`, plus `predict(model_name, payload)`
- `schema.py` — column, type, and range checks
- `input_schema.json` — generated at training time; lists every input field
- `example_request.json` — sample request and response
- `../artifacts/<model>/model.joblib` — fitted sklearn pipeline (preprocessing included)
- `../artifacts/<model>/model_card.json` — features, exclusions, train size, library versions

## Versions

Install the pins in `../requirements.txt`. The artifacts were trained with:

- scikit-learn `1.9.1`
- xgboost `3.4.1`
- pandas `3.0.1`
- numpy `2.3.2`
- joblib `1.6.0`

Mismatch of scikit-learn/xgboost versions is the usual load failure.

## Call it

From the `lake-bloom-severity/` directory (or with that directory on `PYTHONPATH`):

```python
from serving.predictor import predict, predict_xgboost

payload = {
    "tmax_c": 30.1,
    "tmin_c": 17.0,
    "prcp_mm": 0.0,
    "wind_ms": 2.4,
    "rain_sum_7d_mm": 6.9,
    "tmax_mean_7d_c": 31.3,
    "B02": 0.08,
    "B03": 0.13,
    "B04": 0.08,
    "B05": 0.13,
    "B08": 0.06,
    "NDCI": 0.21,
    "NDVI": -0.18,
    "NDWI": 0.38,
    "n_water_px": 678.0,
    "s2_days_offset": -4.0,
    "s2_cloud_pct": 4.6,
    "month": 8,
}

result = predict_xgboost(payload)
# or: predict("random_forest", payload)
```

Batch:

```python
predict("xgboost", {"instances": [payload, payload]})
```

Response shape:

```json
{
  "model": "xgboost",
  "model_version": "1.0.0",
  "predicted_severity": 3,
  "probabilities": {"1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}
}
```

`predicted_severity` is one of `1, 2, 3, 4`.

## Input fields (weather + satellite + month)

Do **not** send `year`, `distance_to_water_m`, `region`, or `data_provider`. Those are excluded from training and rejected at validation.

Required numeric fields are listed in `input_schema.json`.

## Direct joblib load

```python
import joblib
from serving.schema import validate_payload

pipe = joblib.load("artifacts/xgboost/model.joblib")
frame = validate_payload(payload)
pred = pipe.predict(frame)
proba = pipe.predict_proba(frame)
```
