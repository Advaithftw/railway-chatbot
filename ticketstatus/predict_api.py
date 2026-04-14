from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
dataset = pd.read_csv("railway_dataset.csv")


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

@app.post("/predict")
def predict(data: dict):
    """
    Predict confirmation status for a specific train ticket.
    
    Required fields:
    - train_number (int): Train number
    - train_type (str): Type of train (e.g., Shatabdi, Express, Superfast)
    - Other fields: Waitlist Position, Holiday or Peak Season, Seat Availability, Travel Distance, etc.
    """
    df = pd.DataFrame([data])

    # 🔥 Apply same encoding as training
    df = pd.get_dummies(df)

    # 🔥 Add missing columns
    for col in columns:
        if col not in df:
            df[col] = 0

    # 🔥 Ensure correct order
    df = df[columns]

    prob = model.predict_proba(df)[0][1]
    pnr = str(data.get("pnr")).strip() if data.get("pnr") else None
    
    train_info = f"Train {data.get('train_number', 'Unknown')} ({data.get('train_type', 'Unknown')})"

    return {
        "train": train_info,
        "pnr": pnr,
        "probability": float(prob),
        "prediction": "Confirmed" if prob > 0.5 else "Not Confirmed"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")