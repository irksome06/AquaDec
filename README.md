# AquaDec: Lake Bloom Froth Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-2c3e50.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AquaDec is a machine-learning project for classifying cyanobacteria bloom severity in inland waters. The maintained modeling pipeline is in [`lake-bloom-severity/`](lake-bloom-severity/). It trains and evaluates four classifiers using weather observations, Sentinel-2 measurements, image-quality fields, and the observation month.

## Table of Contents

- [Quick Start](#quick-start)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Results](#results)
- [Team Structure](#team-structure)
- [Docker](#docker)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [FAQ](#faq)
- [License](#license)
- [Contact](#contact)

## Quick Start

```powershell
git clone https://github.com/irksome06/AquaDec.git
cd AquaDec/lake-bloom-severity
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocessing/preprocess.py
python run_pipeline.py --models logistic_regression svm random_forest xgboost
```

## Problem Statement

Cyanobacterial blooms can reduce water quality and create health risks for people and animals. AquaDec predicts one of four bloom-severity classes from routinely available weather and satellite observations, supporting analysis and early-warning workflows.

## Dataset

The committed processed table contains 3,688 observations and 24 columns: the `severity4` target, location and metadata columns, and 18 model features. The model features combine weather measurements, Sentinel-2 bands and indices, image-quality fields, and month. The target classes contain 1,255 severity-1, 738 severity-2, 892 severity-3, and 803 severity-4 observations.

The source join combines in-situ bloom observations with NOAA weather and Sentinel-2 measurements. The raw joined file is not committed to this project. To rebuild the processed table, place `caml_weather_sat.csv` in `lake-bloom-severity/data/raw/`.

## Project Structure

The `lake-bloom-severity/` project contains:

- Preprocessing and a frozen location-based train/test split.
- Logistic regression, support vector machine (SVM), random forest, and XGBoost training pipelines.
- Saved model artifacts, model cards, evaluation metrics, and comparison plots.
- A Python prediction interface and input schema for backend integration.
- Configuration files, documentation, notebooks, and tests.

See the [project README](lake-bloom-severity/README.md) for detailed instructions and results. The [backend integration guide](lake-bloom-severity/serving/BACKEND_README.md) documents prediction inputs and responses.

## Installation

The maintained pipeline requires Python 3.8 or newer. From `lake-bloom-severity/`, create an environment and install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

### Data preprocessing

Use the committed processed table as-is, or rebuild it from the raw join:

```powershell
python preprocessing/preprocess.py
```

The output is `data/processed/clean_data.csv`. Preprocessing filters invalid satellite and weather rows, maps severity 5 into severity 4, derives month, removes leakage and identifier columns, and fills wind gaps.

### Training models

Train and evaluate all four implemented models:

```powershell
python run_pipeline.py --models logistic_regression svm random_forest xgboost
```

Run one model without hyperparameter tuning:

```powershell
python run_pipeline.py --models xgboost --skip-tuning
```

Saved artifacts are written to `artifacts/<model>/model.joblib`.

### Running the API

This repository supplies Python prediction helpers but does not include an HTTP server. A FastAPI adapter can expose them at `/predict_stream`. With an adapter running on port 8000, send a request such as:

```bash
curl -N -X POST http://localhost:8000/predict_stream -H "Content-Type: application/json" -d '{"prompt":"{\"tmax_c\":30.1,\"tmin_c\":17.0,\"prcp_mm\":0.0,\"wind_ms\":2.4,\"rain_sum_7d_mm\":6.9,\"tmax_mean_7d_c\":31.3,\"B02\":0.08,\"B03\":0.13,\"B04\":0.08,\"B05\":0.13,\"B08\":0.06,\"NDCI\":0.21,\"NDVI\":-0.18,\"NDWI\":0.38,\"n_water_px\":678.0,\"s2_days_offset\":-4.0,\"s2_cloud_pct\":4.6,\"month\":8}"}'
```

The response contains `predicted_severity` and class probabilities, for example `{"model":"xgboost","predicted_severity":3,"probabilities":{"1":0.1,"2":0.2,"3":0.6,"4":0.1}}`.

### Making predictions

Call a saved pipeline directly from Python:

```python
from serving.predictor import predict_xgboost

payload = {
    "tmax_c": 30.1, "tmin_c": 17.0, "prcp_mm": 0.0, "wind_ms": 2.4,
    "rain_sum_7d_mm": 6.9, "tmax_mean_7d_c": 31.3,
    "B02": 0.08, "B03": 0.13, "B04": 0.08, "B05": 0.13, "B08": 0.06,
    "NDCI": 0.21, "NDVI": -0.18, "NDWI": 0.38, "n_water_px": 678.0,
    "s2_days_offset": -4.0, "s2_cloud_pct": 4.6, "month": 8,
}
print(predict_xgboost(payload))
```

Use `predict_logistic_regression`, `predict_svm`, `predict_random_forest`, or `predict(model_name, payload)` for the other models.

## Data and model inputs

The committed processed dataset is `lake-bloom-severity/data/processed/clean_data.csv`. The original joined raw dataset is not included in that project; see [`lake-bloom-severity/data/raw/README.md`](lake-bloom-severity/data/raw/README.md) for rebuilding instructions.

The prediction interface expects weather, satellite, image-quality, and month fields listed in [`input_schema.json`](lake-bloom-severity/serving/input_schema.json). The models exclude `year`, `distance_to_water_m`, `region`, and `data_provider` from inference.

## Architecture

The pipeline runs preprocessing, location-aware splitting, optional tuning, training, evaluation, and comparison. In a deployed application, an API gateway can fan out a request to the model predictors, collect severity probabilities, and return model results for ranking or presentation.

```text
Client -> API gateway -> model fan-out -> scoring and aggregation -> response
                         |-> Logistic regression
                         |-> SVM
                         |-> Random forest
                         `-> XGBoost
```

Each `.joblib` artifact is a complete pipeline, including feature transformations and its classifier. The XGBoost artifact also needs the project `model` package importable when it is loaded.

## Model Performance

The models were evaluated on the same frozen location-based test split. Recorded results are:

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Logistic regression | 43.4% | 43.4% | 0.424 |
| SVM | 54.1% | 51.5% | 0.506 |
| Random forest | 52.7% | 52.4% | 0.519 |
| XGBoost | **54.8%** | **54.8%** | **0.543** |

These figures describe the committed evaluation split; they are not a guarantee of performance on new locations or future observations. The repository comparison file reports held-out balanced accuracy rather than a separate CV score. Additional per-model metrics and plots are available under `lake-bloom-severity/results/`.

## Results

- XGBoost is the strongest committed model by accuracy, balanced accuracy, and macro F1.
- The fixed location split gives every model the same evaluation basis.
- Weather, Sentinel-2, image-quality, and month features are used; known leakage and metadata fields are excluded.

## Team Structure

| Role | Members | Responsibility |
| --- | --- | --- |
| Project team | AquaDec contributors | Dataset preparation, modeling, evaluation, and integration |
| ML development | Pipeline contributors | Training, tuning, and artifact maintenance |
| Backend integration | Backend team | API adapter and service deployment |
| Documentation | Project maintainers | Reproducibility, tests, and documentation |

## Docker

The repository does not currently include a Dockerfile. When one is added for the API adapter, use:

```bash
docker build -t aquadec-lake-bloom .
docker run --rm -p 8000:8000 aquadec-lake-bloom
```

## Testing

Run the tests from the `lake-bloom-severity/` directory:

```powershell
pytest
```

The project contribution guidance is in [`lake-bloom-severity/CONTRIBUTING.md`](lake-bloom-severity/CONTRIBUTING.md).

## Future Improvements

- Add a versioned HTTP service and health endpoint around the serving helpers.
- Add probability calibration and operational alert thresholds.
- Evaluate temporal and geographic drift on newer observations.
- Monitor input ranges, missing values, and prediction distributions.
- Add a Dockerfile and deployment configuration for the backend adapter.
- Expand integration tests between the frontend, API, and model-serving contract.

## FAQ

**Which model should I use?** XGBoost has the best recorded result in the committed comparison, but all four artifacts are available.

**Does the repository include an HTTP API?** No. It includes Python prediction helpers; a backend service can wrap them.

**What does the model return?** A severity class from 1 through 4 and probabilities for all four classes when supported.

**Can I rebuild the processed dataset?** Yes. Place `caml_weather_sat.csv` in `data/raw/` and run `preprocessing/preprocess.py`.

**Are `year` and `region` model inputs?** No. They are excluded from training and rejected by the serving schema.

## License

This project is intended to be released under the MIT License. Add the repository `LICENSE` file before publishing a formal license grant.

## Contact

For project questions, open an issue or contact the AquaDec maintainers through the [GitHub repository](https://github.com/irksome06/AquaDec). Add the team’s preferred name, email address, and institution when those details are finalized.
