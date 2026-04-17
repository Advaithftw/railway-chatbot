from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from pathlib import Path
import os

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"
DATASET_PATH = BASE_DIR / "railway_dataset.csv"
METRICS_PATH = BASE_DIR / "metrics.json"

model = None
columns = None
dataset = None
_artifact_state = {"model_mtime": None, "columns_mtime": None}


def load_artifacts(force: bool = False):
    global model, columns, dataset, _artifact_state

    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    columns_mtime = COLUMNS_PATH.stat().st_mtime if COLUMNS_PATH.exists() else None

    should_reload = force or model is None or columns is None or dataset is None
    should_reload = should_reload or model_mtime != _artifact_state["model_mtime"]
    should_reload = should_reload or columns_mtime != _artifact_state["columns_mtime"]

    if not should_reload:
        return

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    _artifact_state = {
        "model_mtime": model_mtime,
        "columns_mtime": columns_mtime,
    }


load_artifacts(force=True)


def normalize_travel_class(value):
    text = str(value or "").strip().lower().replace(" ", "")
    if text in {"2ac", "a2"}:
        return "2AC"
    if text in {"3ac", "a3"}:
        return "3AC"
    if text in {"sl", "sleeper"}:
        return "Sleeper"
    if text in {"1ac", "a1"}:
        return "1AC"
    return str(value or "3AC").strip() or "3AC"


def build_default_input_row(df: pd.DataFrame) -> dict:
    working = df.copy()

    if "Waitlist Position" in working.columns:
        working["Waitlist Position"] = (
            working["Waitlist Position"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float)
        )

    if "Holiday or Peak Season" in working.columns:
        working["Holiday or Peak Season"] = working["Holiday or Peak Season"].map({
            "Yes": 1,
            "No": 0,
            True: 1,
            False: 0
        })

    if "Class of Travel" in working.columns:
        working["Class of Travel"] = working["Class of Travel"].map(normalize_travel_class)

    drop_cols = ["PNR Number", "Date of Journey", "Booking Date", "Current Status", "Confirmation Status"]
    working = working.drop(columns=drop_cols, errors="ignore")

    defaults = {}
    for col in working.columns:
        series = working[col].dropna()
        if series.empty:
            defaults[col] = 0
            continue

        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            mode = series.mode()
            defaults[col] = str(mode.iloc[0]) if not mode.empty else str(series.iloc[0])

    # Strong safe defaults for frequent inference fields
    defaults["Class of Travel"] = normalize_travel_class(defaults.get("Class of Travel", "3AC"))
    defaults["Holiday or Peak Season"] = int(float(defaults.get("Holiday or Peak Season", 0)))
    defaults["Waitlist Position"] = float(defaults.get("Waitlist Position", 50))

    return defaults


DEFAULT_INPUT_ROW = build_default_input_row(dataset)


def normalize_train_number(value):
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or text


def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def waitlist_heuristic_probability(data: dict) -> float:
    wl = max(0.0, to_float(data.get("Waitlist Position"), 50.0))
    seats = max(0.0, to_float(data.get("Seat Availability"), 0.0))
    holiday = to_float(data.get("Holiday or Peak Season"), 0.0)
    travel_class = normalize_travel_class(data.get("Class of Travel", "3AC"))

    # Base expectation by waitlist bucket (domain heuristic)
    if wl <= 20:
        base = 0.90
    elif wl <= 45:
        base = 0.72
    elif wl <= 80:
        base = 0.46
    else:
        base = 0.20

    # Seat availability improves confirmation chance
    seat_boost = min(0.20, seats / 600.0)

    # Peak season slightly hurts confirmation probability
    holiday_penalty = 0.08 if holiday >= 1 else 0.0

    class_adjustment = {
        "1AC": 0.08,
        "2AC": 0.04,
        "3AC": 0.0,
        "Sleeper": -0.05
    }.get(travel_class, 0.0)

    calibrated = base + seat_boost - holiday_penalty + class_adjustment
    return max(0.01, min(0.99, calibrated))


def normalize_input_payload(data: dict) -> dict:
    payload = dict(DEFAULT_INPUT_ROW)
    payload.update(data or {})

    # Alias mapping from JS payloads
    if payload.get("train_number") is not None and payload.get("Train Number") is None:
        payload["Train Number"] = payload.get("train_number")
    if payload.get("train_type") and not payload.get("Train Type"):
        payload["Train Type"] = payload.get("train_type")

    payload["Class of Travel"] = normalize_travel_class(payload.get("Class of Travel", "3AC"))
    payload["Waitlist Position"] = max(0.0, to_float(payload.get("Waitlist Position"), 50.0))
    payload["Holiday or Peak Season"] = 1 if str(payload.get("Holiday or Peak Season", 0)).strip().lower() in {"1", "yes", "true"} else 0

    if payload.get("Train Number") is not None:
        train_no = normalize_train_number(payload.get("Train Number"))
        payload["Train Number"] = int(train_no) if str(train_no).isdigit() else payload.get("Train Number")

    return payload


def encode_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    load_artifacts()
    encoded = pd.get_dummies(df)
    for col in columns:
        if col not in encoded:
            encoded[col] = 0
    return encoded[columns]


def build_eval_dataframe() -> pd.DataFrame:
    eval_df = dataset.copy()

    eval_df["Waitlist Position"] = (
        eval_df["Waitlist Position"].astype(str).str.extract(r"(\d+)")[0]
    )
    eval_df["Waitlist Position"] = pd.to_numeric(eval_df["Waitlist Position"], errors="coerce")

    eval_df["Holiday or Peak Season"] = eval_df["Holiday or Peak Season"].map({
        "Yes": 1,
        "No": 0,
        True: 1,
        False: 0
    })

    eval_df["Confirmation Status"] = eval_df["Confirmation Status"].map({
        "Confirmed": 1,
        "Not Confirmed": 0
    })

    eval_df = eval_df.drop(columns=[
        "PNR Number",
        "Date of Journey",
        "Booking Date",
        "Current Status"
    ], errors="ignore")

    eval_df = eval_df.fillna({
        "Seat Availability": 0,
        "Travel Distance": eval_df["Travel Distance"].mean(),
        "Waitlist Position": eval_df["Waitlist Position"].median(),
    })

    eval_df["Class of Travel"] = eval_df["Class of Travel"].map(normalize_travel_class)
    eval_df = eval_df.dropna(subset=["Confirmation Status"])

    return eval_df


def detect_leakage(raw_df: pd.DataFrame) -> dict:
    y = raw_df["Confirmation Status"].map({
        "Confirmed": 1,
        "Not Confirmed": 0
    })

    wl_series = pd.to_numeric(
        raw_df["Waitlist Position"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    valid_mask = y.notna()
    y = y[valid_mask]
    wl_series = wl_series[valid_mask]

    wl_presence_by_target = (
        pd.DataFrame({"target": y, "wl_present": wl_series.notna().astype(int)})
        .groupby("target")["wl_present"]
        .mean()
        .to_dict()
    )

    confirmed_presence = float(wl_presence_by_target.get(1, 0.0))
    not_confirmed_presence = float(wl_presence_by_target.get(0, 0.0))
    separation_gap = abs(confirmed_presence - not_confirmed_presence)

    return {
        "waitlist_presence_confirmed": confirmed_presence,
        "waitlist_presence_not_confirmed": not_confirmed_presence,
        "waitlist_presence_gap": separation_gap,
        "high_risk": separation_gap >= 0.95,
    }


def compute_holdout_metrics(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            min_samples_split=4,
            max_features="sqrt",
            n_jobs=-1,
        ),
    }

    best_name = None
    best_model = None
    best_threshold = 0.5
    best_val_f1 = -1

    for model_name, candidate in candidates.items():
        candidate.fit(x_fit, y_fit)
        val_prob = candidate.predict_proba(x_val)[:, 1]

        local_best_f1 = -1
        local_best_threshold = 0.5
        for threshold in np.arange(0.30, 0.71, 0.02):
            val_pred = (val_prob >= threshold).astype(int)
            score = f1_score(y_val, val_pred, zero_division=0)
            if score > local_best_f1:
                local_best_f1 = score
                local_best_threshold = float(threshold)

        if local_best_f1 > best_val_f1:
            best_val_f1 = local_best_f1
            best_name = model_name
            best_model = candidate
            best_threshold = local_best_threshold

    best_model.fit(x_train, y_train)
    test_prob = best_model.predict_proba(x_test)[:, 1]
    y_pred = (test_prob >= best_threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "samples": int(len(y_test)),
        "model_name": best_name,
        "threshold": float(best_threshold),
    }

@app.post("/predict")
def predict(data: dict):
    """
    Predict confirmation status for a specific train ticket.
    
    Required fields:
    - train_number (int): Train number
    - train_type (str): Type of train (e.g., Shatabdi, Express, Superfast)
    - Other fields: Waitlist Position, Holiday or Peak Season, Seat Availability, Travel Distance, etc.
    """
    load_artifacts()
    payload = normalize_input_payload(data)
    df = pd.DataFrame([payload])

    df = encode_features_for_model(df)

    model_prob = float(model.predict_proba(df)[0][1])
    heuristic_prob = waitlist_heuristic_probability(payload)
    prob = (0.7 * model_prob) + (0.3 * heuristic_prob)

    # Guardrail for WL cases, but preserve ranking (WL 15 > WL 45).
    wl = max(0.0, to_float(payload.get("Waitlist Position"), 50.0))
    if wl <= 60:
        # Dynamic floor decreases as WL increases.
        # Examples: WL15 -> 0.655, WL45 -> 0.565, WL60 -> 0.520
        cls = normalize_travel_class(payload.get("Class of Travel", "3AC"))
        floor_class_adjustment = {
            "1AC": 0.03,
            "2AC": 0.015,
            "3AC": 0.0,
            "Sleeper": -0.02
        }.get(cls, 0.0)

        wl_floor = (0.70 - (0.003 * wl)) + floor_class_adjustment
        prob = max(prob, wl_floor)

    prob = max(0.01, min(0.99, float(prob)))
    pnr = str(data.get("pnr")).strip() if data.get("pnr") else None

    train_number = payload.get("Train Number") or payload.get("train_number") or "Unknown"
    train_type = payload.get("Train Type") or payload.get("train_type") or "Unknown"
    train_info = f"Train {train_number} ({train_type})"

    return {
        "train": train_info,
        "pnr": pnr,
        "probability": float(prob),
        "prediction": "Confirmed" if prob > 0.5 else "Not Confirmed",
        "model_probability": model_prob,
        "heuristic_probability": heuristic_prob
    }


@app.get("/metrics")
def metrics():
    load_artifacts()

    eval_df = build_eval_dataframe()
    y_true = eval_df["Confirmation Status"]
    baseline = float(max(y_true.mean(), 1 - y_true.mean())) if len(y_true) else 0.0
    leakage = detect_leakage(dataset)

    raw_metrics = None
    if METRICS_PATH.exists():
        try:
            import json
            with open(METRICS_PATH, "r") as f:
                saved_metrics = json.load(f)
            raw_metrics = {
                "accuracy": float(saved_metrics.get("accuracy", 0.0)),
                "precision": float(saved_metrics.get("precision", 0.0)),
                "recall": float(saved_metrics.get("recall", 0.0)),
                "f1": float(saved_metrics.get("f1", 0.0)),
                "cv_f1_mean": float(saved_metrics.get("cv_f1_mean", 0.0)),
                "cv_f1_std": float(saved_metrics.get("cv_f1_std", 0.0)),
                "baseline_accuracy": float(saved_metrics.get("baseline_accuracy", baseline)),
                "samples": int(saved_metrics.get("test_samples", saved_metrics.get("samples", 0))),
                "source": "metrics.json"
            }
        except Exception:
            raw_metrics = None

    has_perfect_raw = bool(raw_metrics) and all(
        raw_metrics.get(metric, 0.0) >= 0.9999
        for metric in ["accuracy", "precision", "recall", "f1"]
    )

    if leakage["high_risk"] or has_perfect_raw:
        x_safe = eval_df.drop(columns=["Confirmation Status", "Waitlist Position"], errors="ignore")
        x_safe = pd.get_dummies(x_safe)
        robust = compute_holdout_metrics(x_safe, y_true)

        return {
            "accuracy": robust["accuracy"],
            "precision": robust["precision"],
            "recall": robust["recall"],
            "f1": robust["f1"],
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
            "baseline_accuracy": baseline,
            "samples": robust["samples"],
            "source": "leakage_robust_holdout",
            "model_name": robust.get("model_name"),
            "threshold": robust.get("threshold"),
            "warning": "Potential target leakage detected in dataset. Showing leakage-robust holdout metrics (without Waitlist Position) instead of optimistic raw metrics.",
            "leakage": leakage,
            "raw_metrics": raw_metrics,
        }

    x_raw = eval_df.drop(columns=["Confirmation Status"])
    x_eval = encode_features_for_model(x_raw)

    y_pred = model.predict(x_eval)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cv_f1_mean": float(0.0),
        "cv_f1_std": float(0.0),
        "baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "samples": int(len(y_true)),
        "source": "live_recompute"
    }


@app.post("/reload")
def reload_model():
    load_artifacts(force=True)
    return {
        "status": "reloaded",
        "model_path": str(MODEL_PATH),
        "columns_path": str(COLUMNS_PATH),
        "metrics_path": str(METRICS_PATH),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")