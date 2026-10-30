"""Train the Random Forest for lake bloom severity and save it as a .pkl file.

Run:  python train_rf.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV = "data/processed/clean_data.csv"   # <- change to your CSV path
MODEL_OUT = "rf_model.pkl"
SEED = 42

FEATURES = (
    ["data_provider", "region", "distance_to_water_m", "year", "month"]          # original
    + ["tmax_c", "tmin_c", "prcp_mm", "wind_ms", "rain_sum_7d_mm", "tmax_mean_7d_c"]  # weather
    + ["B02", "B03", "B04", "B05", "B08", "NDCI", "NDVI", "NDWI"]                # satellite
    + ["n_water_px", "s2_days_offset", "s2_cloud_pct"]                           # quality
)
CATEGORICAL = ["data_provider", "region"]

# ---- data: keep each lake entirely in train OR test ------------------------
df = pd.read_csv(CSV)
X, y, groups = df[FEATURES], df["severity4"], df["location_group"]

train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

# ---- model: one-hot encode text columns + Random Forest --------------------
model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough")),
    ("rf", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED)),
])
model.fit(X_train, y_train)

# ---- save ------------------------------------------------------------------
joblib.dump(model, MODEL_OUT)
print(f"trained on {len(train_idx)} rows, saved to {MODEL_OUT}")

# To load later:
#   model = joblib.load("rf_model.pkl")
#   model.predict(new_df[FEATURES])
