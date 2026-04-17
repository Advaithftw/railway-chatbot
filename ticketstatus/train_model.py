import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json
import numpy as np

print("🚀 Loading dataset...")
df = pd.read_csv("railway_dataset.csv")

print("🧹 Cleaning data...")

# -----------------------------
# FIX WAITLIST
# -----------------------------
df["Waitlist Position"] = df["Waitlist Position"].astype(str)
df["Waitlist Position"] = df["Waitlist Position"].str.extract(r'(\d+)')
df["Waitlist Position"] = pd.to_numeric(df["Waitlist Position"], errors="coerce")

# -----------------------------
# FIX HOLIDAY COLUMN
# -----------------------------
df["Holiday or Peak Season"] = df["Holiday or Peak Season"].map({
    "Yes": 1,
    "No": 0,
    True: 1,
    False: 0
})

# -----------------------------
# TARGET
# -----------------------------
df["Confirmation Status"] = df["Confirmation Status"].map({
    "Confirmed": 1,
    "Not Confirmed": 0
})

# -----------------------------
# DROP USELESS / LEAKAGE
# 🔥 Keep Train Number & Train Type for train-specific predictions
# -----------------------------
df = df.drop(columns=[
    "PNR Number",
    "Date of Journey",
    "Booking Date",
    "Current Status"   # 🔥 important
], errors="ignore")

# -----------------------------
# DROP LEAKAGE-PRONE FEATURES
# Waitlist fields are label-proxy in this synthetic dataset
# -----------------------------
df = df.drop(columns=[
    "Waitlist Position",
    "Waitlist Missing"
], errors="ignore")

# -----------------------------
# DROP NULLS
# -----------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    if col == "Confirmation Status":
        continue
    median_value = df[col].median()
    if pd.isna(median_value):
        median_value = 0
    df[col] = df[col].fillna(median_value)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# 🔥 AUTO ENCODE EVERYTHING
# -----------------------------
X = df.drop(columns=["Confirmation Status"])
y = df["Confirmation Status"]

# Automatically convert ALL object columns
X = pd.get_dummies(X)

# -----------------------------
# DEBUG CHECK
# -----------------------------
print("🔍 Remaining object columns:")
print(X.select_dtypes(include=["object"]).columns)

print("📊 Data shape:", X.shape)

# Keep train/test alignment deterministic
X = X.reindex(sorted(X.columns), axis=1)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Validation split for threshold tuning/model selection
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# -----------------------------
# TRAIN WITH PROGRESS BAR
# -----------------------------
print("🤖 Training model...")

model_candidates = {
    "random_forest": RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        max_depth=14,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=160,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
        min_samples_split=4,
        max_features="sqrt",
        max_depth=14,
        n_jobs=-1,
    ),
}

best_model_name = None
best_model = None
best_threshold = 0.5
best_val_f1 = -1

for model_name, candidate in model_candidates.items():
    candidate.fit(X_fit, y_fit)
    val_prob = candidate.predict_proba(X_val)[:, 1]

    local_best_f1 = -1
    local_best_threshold = 0.5
    for threshold in np.arange(0.30, 0.71, 0.02):
        val_pred = (val_prob >= threshold).astype(int)
        score = f1_score(y_val, val_pred, zero_division=0)
        if score > local_best_f1:
            local_best_f1 = score
            local_best_threshold = float(threshold)

    print(f"🔎 {model_name}: best validation F1={local_best_f1:.4f} at threshold={local_best_threshold:.2f}")

    if local_best_f1 > best_val_f1:
        best_val_f1 = local_best_f1
        best_model_name = model_name
        best_model = candidate
        best_threshold = local_best_threshold

print(f"✅ Selected model: {best_model_name} (threshold={best_threshold:.2f})")

# Refit selected model on full training partition
best_model.fit(X_train, y_train)

# -----------------------------
# CROSS-VALIDATION (more credible than a single holdout)
# -----------------------------
print("🔁 Running stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)

majority_baseline = max(y.mean(), 1 - y.mean())

# -----------------------------
# EVALUATE
# -----------------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred).tolist()

print(f"🎯 Model accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")
print(f"🎯 CV F1 Mean: {cv_scores.mean():.4f}")
print(f"🎯 CV F1 Std: {cv_scores.std():.4f}")
print(f"🎯 Majority Baseline Accuracy: {majority_baseline:.4f}")
print(f"🎯 Validation-selected threshold: {best_threshold:.2f}")
print("🧾 Confusion Matrix:")
print(cm)
print("📄 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "cv_f1_mean": float(cv_scores.mean()),
    "cv_f1_std": float(cv_scores.std()),
    "baseline_accuracy": float(majority_baseline),
    "model_name": best_model_name,
    "threshold": float(best_threshold),
    "confusion_matrix": cm,
    "test_samples": int(len(y_test)),
    "train_samples": int(len(y_train)),
}

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Model trained successfully!")