"""
decision_tree_model.py
======================
Lake Bloom Severity Prediction using Decision Tree Classification.

This script:
  1. Loads the preprocessed dataset (clean_data.csv).
  2. Splits data using GroupShuffleSplit (location-based) to prevent data leakage.
  3. Builds a scikit-learn Pipeline with ColumnTransformer + DecisionTreeClassifier.
  4. Trains, predicts, and evaluates the model.
  5. Performs hyperparameter tuning with GridSearchCV (GroupKFold).
  6. Generates publication-quality plots (confusion matrix, feature importances, tree).
  7. Prints a concise model explanation.

Author : [Student Name]
Date   : 2026-10-24
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

import sys
# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress non-critical warnings for clean output
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "clean_data.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data (1).csv"),
]
DATA_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "severity4"
GROUP_COL = "location_group"

# Columns that must NEVER be used as features (leak target or are IDs)
EXCLUDE_COLS = {TARGET_COL, GROUP_COL, "severity", "abundance"}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Separate features, target, and groups
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 2: Preparing features (X), target (y), and groups")
print("=" * 70)

# Drop any columns that exist in EXCLUDE_COLS
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X = df[feature_cols]
y = df[TARGET_COL]
groups = df[GROUP_COL]

print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"  Target column   : {TARGET_COL}")
print(f"  Group column    : {GROUP_COL}")
print(f"\n  Target distribution:")
print(y.value_counts().sort_index().to_string())
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Identify categorical vs numeric columns
# ──────────────────────────────────────────────────────────────────────────────
# Categorical columns: object / category dtypes
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
# Numeric columns: everything else
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"  Numeric columns     ({len(num_cols)}): {num_cols}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train-test split using GroupShuffleSplit
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 3: Train-test split (GroupShuffleSplit)")
print("=" * 70)

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print(f"  Train set : {X_train.shape}  |  Test set : {X_test.shape}")
print(f"  Train target distribution:")
print(y_train.value_counts().sort_index().to_string())
print(f"\n  Test target distribution:")
print(y_test.value_counts().sort_index().to_string())
print()

# Verify no group leakage
train_groups = set(groups_train.unique())
test_groups = set(groups_test.unique())
overlap = train_groups & test_groups
print(f"  Unique groups in train: {len(train_groups)}")
print(f"  Unique groups in test : {len(test_groups)}")
print(f"  Group overlap         : {len(overlap)} (should be 0)")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build scikit-learn Pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 4: Building Pipeline (ColumnTransformer + DecisionTreeClassifier)")
print("=" * 70)

# Preprocessor: one-hot encode categoricals, passthrough numerics
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Decision tree with specified hyperparameters
dt_clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

# Full pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", dt_clf),
    ]
)

print("  Pipeline structure:")
print(f"    {pipeline}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Train the pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 5: Training the pipeline")
print("=" * 70)

pipeline.fit(X_train, y_train)
print("  Training complete.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Predict on test data
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 6: Predictions & Evaluation")
print("=" * 70)

y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

# ── Evaluation metrics ────────────────────────────────────────────────────────
severity_labels = sorted(y.unique())

print("\n  -- Classification Report (Test Set) --")
print(
    classification_report(
        y_test, y_pred_test, target_names=[f"Class {c}" for c in severity_labels], digits=3
    )
)

cm = confusion_matrix(y_test, y_pred_test, labels=severity_labels)

macro_f1_test = f1_score(y_test, y_pred_test, average="macro")
bal_acc_test = balanced_accuracy_score(y_test, y_pred_test)
acc_test = accuracy_score(y_test, y_pred_test)

macro_f1_train = f1_score(y_train, y_pred_train, average="macro")
bal_acc_train = balanced_accuracy_score(y_train, y_pred_train)
acc_train = accuracy_score(y_train, y_pred_train)

print(f"  Macro F1 Score      (test) : {macro_f1_test:.4f}")
print(f"  Balanced Accuracy   (test) : {bal_acc_test:.4f}")
print(f"  Overall Accuracy    (test) : {acc_test:.4f}")
print()
print(f"  Macro F1 Score      (train): {macro_f1_train:.4f}")
print(f"  Balanced Accuracy   (train): {bal_acc_train:.4f}")
print(f"  Overall Accuracy    (train): {acc_train:.4f}")
print()

print("  ── Confusion Matrix ──")
print(f"  Labels: {severity_labels}")
print(cm)
print()

# ──────────────────────────────────────────────────────────────────────────────
# 8. Plot confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 7: Plotting confusion matrix")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"Class {c}" for c in severity_labels],
    yticklabels=[f"Class {c}" for c in severity_labels],
    ax=ax,
    linewidths=0.5,
    linecolor="gray",
)
ax.set_xlabel("Predicted Severity", fontsize=12)
ax.set_ylabel("Actual Severity", fontsize=12)
ax.set_title("Confusion Matrix – Decision Tree (Severity4)", fontsize=14)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  Saved → {cm_path}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 9. Feature importances (top 15)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 8: Feature importances (top 15)")
print("=" * 70)

# Get feature names after one-hot encoding
ohe_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_

# Build a DataFrame and sort
feat_imp_df = (
    pd.DataFrame({"Feature": ohe_feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

print(feat_imp_df.to_string(index=False))
print()

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 7))
feat_imp_sorted = feat_imp_df.sort_values("Importance", ascending=True)  # ascending for barh
bars = ax.barh(
    feat_imp_sorted["Feature"],
    feat_imp_sorted["Importance"],
    color=sns.color_palette("viridis", n_colors=15),
    edgecolor="black",
    linewidth=0.5,
)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.set_title("Top 15 Feature Importances – Decision Tree", fontsize=14)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importances.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"  Saved → {fi_path}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 10. Plot decision tree (first 3 levels)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 9: Plotting decision tree (first 3 levels)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(28, 12))
plot_tree(
    pipeline.named_steps["classifier"],
    max_depth=3,
    feature_names=ohe_feature_names,
    class_names=[f"Class {c}" for c in severity_labels],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    proportion=True,
)
ax.set_title("Decision Tree – First 3 Levels", fontsize=16)
plt.tight_layout()
tree_path = os.path.join(OUTPUT_DIR, "decision_tree_3levels.png")
plt.savefig(tree_path, dpi=150)
plt.close()
print(f"  Saved → {tree_path}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 11. Hyperparameter Tuning with GridSearchCV (GroupKFold)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 10: Hyperparameter Tuning (GridSearchCV + GroupKFold)")
print("=" * 70)

param_grid = {
    "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
    "classifier__min_samples_leaf": [5, 10, 20, 30, 50],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__criterion": ["gini", "entropy"],
}

gkf = GroupKFold(n_splits=5)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=gkf,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

# Fit on training data only, passing the corresponding location groups
print("  Fitting GridSearchCV (this may take a while)...\n")
grid_search.fit(X_train, y_train, groups=groups_train)

print(f"\n  Best Parameters          : {grid_search.best_params_}")
print(f"  Best CV Macro F1 Score   : {grid_search.best_score_:.4f}")
print()

# Evaluate best model on test set
y_pred_best = grid_search.best_estimator_.predict(X_test)
best_macro_f1 = f1_score(y_test, y_pred_best, average="macro")
best_bal_acc = balanced_accuracy_score(y_test, y_pred_best)
best_acc = accuracy_score(y_test, y_pred_best)

print("  ── Best Model – Test Set Evaluation ──")
print(
    classification_report(
        y_test, y_pred_best, target_names=[f"Class {c}" for c in severity_labels], digits=3
    )
)
print(f"  Macro F1 Score      (test) : {best_macro_f1:.4f}")
print(f"  Balanced Accuracy   (test) : {best_bal_acc:.4f}")
print(f"  Overall Accuracy    (test) : {best_acc:.4f}")
print()

# Also compute train metrics for best model (overfitting check)
y_pred_best_train = grid_search.best_estimator_.predict(X_train)
best_macro_f1_train = f1_score(y_train, y_pred_best_train, average="macro")
best_bal_acc_train = balanced_accuracy_score(y_train, y_pred_best_train)

print(f"  Macro F1 Score      (train): {best_macro_f1_train:.4f}")
print(f"  Balanced Accuracy   (train): {best_bal_acc_train:.4f}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 12. Summary & Explanation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 11: Model Explanation")
print("=" * 70)

explanation = """
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL EXPLANATION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Why GroupShuffleSplit was used:                                  │
│     GroupShuffleSplit ensures that samples from the same location    │
│     (lake) never appear in both the training and test sets. This    │
│     prevents data leakage – without it, the model could memorize   │
│     location-specific patterns and report inflated test accuracy    │
│     that would not generalize to truly unseen lakes.                │
│                                                                     │
│  2. Why one-hot encoding was needed:                                │
│     Columns like 'region' and 'data_provider' are categorical      │
│     (text) variables. Scikit-learn's DecisionTreeClassifier cannot  │
│     handle raw strings. One-hot encoding converts each category    │
│     into a binary (0/1) column, making them usable as numeric      │
│     features without imposing a false ordinal relationship.         │
│                                                                     │
│  3. Why scaling was NOT used:                                       │
│     Decision Trees split features by comparing values to           │
│     thresholds; they are invariant to monotonic transformations     │
│     of the input features. Scaling (e.g., StandardScaler) does     │
│     not change the split points or the resulting tree structure,   │
│     so it is unnecessary and was intentionally omitted.            │
│                                                                     │
│  4. Why class_weight="balanced" was used:                           │
│     The target variable (severity4) is imbalanced: class 1 has     │
│     ~1,255 samples while class 2 has only ~738. Using              │
│     class_weight="balanced" automatically adjusts the weight of    │
│     each class inversely proportional to its frequency, so the     │
│     model pays more attention to minority classes and does not     │
│     simply predict the majority class.                              │
│                                                                     │
│  5. Most important features:                                        │
│     (See the Top 15 Feature Importances table and bar chart above. │
│     These are the features the Decision Tree relied on most for    │
│     splitting the data – higher importance means the feature       │
│     contributed more to reducing Gini impurity / entropy.)          │
│                                                                     │
│  6. Overfitting vs. Underfitting analysis:                          │
│     Compare the train and test Macro F1 / Balanced Accuracy above: │
│     • If train >> test → the model is OVERFITTING                  │
│       (memorizing train data, poor generalization).                 │
│     • If train ≈ test and both are low → UNDERFITTING              │
│       (model is too simple to capture the patterns).               │
│     • If train ≈ test and both are reasonable → good balance.      │
│     The GridSearchCV best model's train vs. test gap provides      │
│     the clearest signal. Check the printed metrics above.           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

print(explanation)

# ── Final quick summary ───────────────────────────────────────────────────────
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset             : {DATA_PATH}")
print(f"  Total samples       : {len(df)}")
print(f"  Total features used : {len(feature_cols)}")
print(f"  Train shape         : {X_train.shape}")
print(f"  Test shape          : {X_test.shape}")
print(f"  Class distribution  : {dict(y.value_counts().sort_index())}")
print()
print("  ── Default Model ──")
print(f"  Train Macro F1      : {macro_f1_train:.4f}")
print(f"  Test  Macro F1      : {macro_f1_test:.4f}")
print(f"  Train Balanced Acc  : {bal_acc_train:.4f}")
print(f"  Test  Balanced Acc  : {bal_acc_test:.4f}")
print()
print("  ── Best Model (GridSearchCV) ──")
print(f"  Best Params         : {grid_search.best_params_}")
print(f"  Best CV Macro F1    : {grid_search.best_score_:.4f}")
print(f"  Train Macro F1      : {best_macro_f1_train:.4f}")
print(f"  Test  Macro F1      : {best_macro_f1:.4f}")
print(f"  Train Balanced Acc  : {best_bal_acc_train:.4f}")
print(f"  Test  Balanced Acc  : {best_bal_acc:.4f}")
print()

# ── Overfitting/underfitting verdict ──────────────────────────────────────────
gap = best_macro_f1_train - best_macro_f1
if gap > 0.15:
    verdict = "OVERFITTING – large gap between train and test performance."
elif best_macro_f1 < 0.40 and best_macro_f1_train < 0.50:
    verdict = "UNDERFITTING – both train and test scores are low."
elif gap < 0.05:
    verdict = "GOOD FIT – train and test scores are close."
else:
    verdict = "SLIGHT OVERFITTING – moderate gap; consider stronger regularization."

print(f"  Model Fit Verdict   : {verdict}")
print()
print(f"  Top 5 Features      :")
for _, row in feat_imp_df.head(5).iterrows():
    print(f"    {row['Feature']:40s}  {row['Importance']:.4f}")
print()
print(f"  All plots saved to  : {OUTPUT_DIR}")
print("=" * 70)
print("Done!")
