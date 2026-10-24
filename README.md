# AquaDec: Lake Bloom Severity Prediction & Risk Early Warning

AquaDec is a machine learning system designed to predict cyanobacteria bloom severity in US inland waters using remote sensing and meteorological observations.

Cyanobacteria bloom severity serves as an early warning proxy for toxic algal events and lake froth formation risk.

---

## 🔬 Dataset Overview

The dataset integrates three heterogeneous sources:
* **In-situ Abundance:** NASA SeaBASS (CAML dataset) with field cyanobacteria abundance across inland US lakes.
* **Daily Meteorology:** NOAA GHCN-Daily stations (maximum/minimum temperature, precipitation, wind speed, 7-day rainfall, 7-day mean temperature).
* **Satellite Observations:** Sentinel-2 multispectral surface reflectance from Microsoft Planetary Computer (bands B02, B03, B04, B05, B08, and indices NDCI, NDVI, NDWI).

The target variable is **`severity4`** (discrete ordinal bloom severity levels 1, 2, 3, 4).

---

## 🌲 Decision Tree Model (`decision_tree_model.py`)

The baseline Decision Tree classifier is constructed with strict spatial independence to prevent geographical memorization and data leakage:

* **Location-Aware Partitioning:** Evaluated using `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on `location_group`. Exactly **0** lake locations overlap between the 1,143 training locations and 286 test locations.
* **Optimal Hyperparameters:** `criterion="entropy"`, `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42`.
* **Preprocessing:** One-hot encoding for categorical variables (`region`, `data_provider`) with unscaled numeric features.

### Decision Tree Performance Metrics

| Metric | Test Set (Held-Out) | Training Set | 5-Fold GroupKFold CV |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **57.29%** | **63.65%** | **58.08%** ($\pm 0.0323$) |
| **Balanced Accuracy**| **56.06%** | **61.48%** | **57.26%** ($\pm 0.0146$) |
| **Macro F1 Score**   | **57.26%** | **61.93%** | **58.08%** ($\pm 0.0128$) |
| **Weighted F1 Score** | **57.16%** | 63.38% | — |

* **Per-Class Metrics:**
  * **Class 1 (Clean):** Precision = 59.9%, Recall = 73.1%, F1 = 65.9%
  * **Class 2 (Moderate):** Precision = 32.9%, Recall = 30.4%, F1 = 31.6%
  * **Class 3 (High):** Precision = 50.5%, Recall = 46.2%, F1 = 48.3%
  * **Class 4 (Severe/Toxic):** Precision = **94.4%**, Recall = **74.5%**, F1 = **83.3%**

---

## 🚀 Modular ML Pipeline Architecture (`lake_bloom_xgboost`)

In addition to the standalone script, the repository includes an end-to-end modular pipeline structure:

```
AquaDec/
├── configs/
│   ├── config.yaml          # Paths, split, feature sets, model and tuning settings
│   ├── evaluation.yaml      # Evaluation metrics, plots and comparison settings
│   └── best_params.yaml     # Tuned parameters written by finetune.py
├── data/
│   ├── raw/                 # Satellite & weather inputs (caml_weather_sat.csv)
│   └── processed/           # Clean dataset (clean_data.csv)
├── preprocessing/
│   └── preprocess.py        # Cleaning: raw -> processed
├── model/
│   ├── utils.py             # Config loader, GroupShuffleSplit, encoding helpers
│   ├── finetune.py          # GroupKFold hyperparameter search
│   ├── train.py             # Final model training -> outputs/models/xgb_model.pkl
│   └── evaluate.py          # Comprehensive evaluation & plot generation
├── outputs/
│   ├── models/              # Serialized model artifacts (xgb_model.pkl)
│   └── reports/             # metrics.json, confusion matrices, feature importances
├── decision_tree_model.py   # Standalone Decision Tree model pipeline
├── model_validation_report.py # Automated validation and audit report script
├── run_pipeline.py          # Master CLI orchestrator
└── requirements.txt
```

### XGBoost Model Performance

* **Test Accuracy:** **62.16%** (vs. 57.29% Decision Tree)
* **Test Macro F1:** **0.6137** (vs. 0.5726 Decision Tree)
* **Test Balanced Accuracy:** **0.6089** (vs. 0.5606 Decision Tree)
* **Majority Baseline Gain:** +27.47% accuracy gain over dummy classifier.

---

## 🛠️ Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Decision Tree Model
```bash
python decision_tree_model.py
```

### 3. Generate the Model Validation Audit Report
```bash
python model_validation_report.py
```

### 4. Run the Full End-to-End Pipeline
```bash
python run_pipeline.py
```
Or skip tuning to train directly:
```bash
python run_pipeline.py --skip-tuning
```
