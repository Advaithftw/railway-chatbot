import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json
from tqdm import tqdm

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
# DROP NULLS
# -----------------------------
df = df.fillna({
    "Seat Availability": 0,
    "Travel Distance": df["Travel Distance"].mean(),
})

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

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TRAIN WITH PROGRESS BAR
# -----------------------------
print("🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# -----------------------------
# CROSS-VALIDATION (more credible than a single holdout)
# -----------------------------
print("🔁 Running stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model,
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
y_pred = model.predict(X_test)

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
    "confusion_matrix": cm,
    "test_samples": int(len(y_test)),
    "train_samples": int(len(y_train)),
}

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Model trained successfully!")