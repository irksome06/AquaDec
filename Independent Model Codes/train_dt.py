"""Train the Decision Tree for lake bloom severity and save it as a .pkl file.

Usage:
    python train_dt.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score

CSV = "data/processed/clean_data.csv"
MODEL_OUT = "dt_model.pkl"
SEED = 42

FEATURES = (
    ["data_provider", "region", "distance_to_water_m", "year", "month"]          # metadata
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
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# ---- model: one-hot encode categoricals + Decision Tree --------------------
model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough")),
    ("dt", DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED)),
])
model.fit(X_train, y_train)

# ---- evaluate on test set --------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Test Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
print(f"Test Balanced Accuracy: {bal_acc:.4f}")
print(f"Test Macro F1         : {macro_f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ---- save ------------------------------------------------------------------
joblib.dump(model, MODEL_OUT)
print(f"Trained on {len(train_idx)} rows, saved to {MODEL_OUT}")
