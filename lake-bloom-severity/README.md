# Lake bloom severity

Four classifiers for cyanobacteria bloom severity (`severity4`: 1–4) from **weather + Sentinel-2** (plus `month` and image-quality fields). Built for a backend `predict` call. This folder is self-contained; the rest of AquaDec is unchanged.

**Not used for training or inference:** `year`, `distance_to_water_m`, `region`, `data_provider`.

## Setup

```bash
cd lake-bloom-severity
pip install -r requirements.txt
```

Processed data is already in `data/processed/clean_data.csv`. Frozen locations are in `data/splits/splits.csv`. Put `caml_weather_sat.csv` in `data/raw/` only if you need to rebuild the processed table.

## Run

```bash
python run_pipeline.py --models logistic_regression svm random_forest xgboost
```

Stages: preprocess → split (skipped if `splits.csv` exists) → finetune → train → evaluate → compare.

```bash
python run_pipeline.py --models xgboost --skip-tuning
python model/train.py --model random_forest
python model/evaluate.py --model random_forest
python model/compare.py
```

## Serving

See [serving/BACKEND_README.md](serving/BACKEND_README.md).

```python
from serving.predictor import predict_xgboost
predict_xgboost({...weather and satellite fields..., "month": 8})
```

## Results (frozen location test)

Same `data/splits/splits.csv` for every model. Features are weather + satellite + month only (no year / distance to water / region / data provider).

| Model | Accuracy | Balanced acc. | Macro F1 |
| :--- | :---: | :---: | :---: |
| logistic_regression | 43.4% | 43.4% | 0.424 |
| svm | 54.1% | 51.5% | 0.506 |
| random_forest | 52.7% | 52.4% | 0.519 |
| xgboost | **54.8%** | **54.8%** | **0.543** |

Plots: `results/<model>/confusion_matrix.png`, `results/comparison/comparison.png`.

## Layout

- `configs/` — shared paths/features/split checksum, evaluation, per-model YAML
- `model/` — load frozen split, pipelines, tune/train/evaluate/compare
- `serving/` — schema + one predict function per model
- `artifacts/` — committed `model.joblib` + `model_card.json`
- `results/` — committed metrics (small)
- `outputs/` — git-ignored large local files
