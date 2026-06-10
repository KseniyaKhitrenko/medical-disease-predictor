from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
FEATURES_PATH = BASE_DIR / "models" / "features.joblib"

app = FastAPI()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def root():
    return {"message": "Heart disease predictor is running"}

@app.post("/predict")
def predict(data: HeartInput):
    df = pd.DataFrame([data.dict()])[features]
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0, 1])
    return {"prediction": pred, "probability": proba}
