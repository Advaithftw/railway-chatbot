import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
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
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN WITH PROGRESS BAR
# -----------------------------
print("🤖 Training model...")

model = RandomForestClassifier(n_estimators=1, warm_start=True)

for i in tqdm(range(100), desc="Training Progress"):
    model.n_estimators = i + 1
    model.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"🎯 Model accuracy: {accuracy:.4f}")

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model trained successfully!")