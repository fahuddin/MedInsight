from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import io
# Import from your modules
from src.features.feature_engineering import feature_engineering
from src.models.train_model import train_and_save_model
from src.models.evaluate import evaluate_model
from src.nlp.text_cleaning import clean_text
from src.nlp.embeddings import generate_embeddings
from src.nlp.ner_extraction import extract_entities
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import List, Optional
import joblib

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_readmission_model.pkl"))
FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "feature_columns.pkl"))



class PatientData(BaseModel):
    ID: int
    Age: int
    BloodPressure: float
    Cholesterol: float
    Glucose: float
    HeartRate: float
    BMI: float
    Smoker: bool
    NumberOfVisits: int
    Readmission: bool
    ClinicalNotes: str
    Gender: str

@app.get("/")
def root():
    return {"message":"Welcome to the MediInsight API"}

def load_artifacts():
    global MODEL, FEATURE_COLUMNS
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURES_PATH):
        FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
    else:
        FEATURE_COLUMNS = None  # fallback: infer runtime but better to save during training
# Load artifacts at startup
app = FastAPI(lifespan=load_artifacts)


class SingleRecord(BaseModel):
    # adjust fields to match expected input schema or accept raw clinical fields
    patient_id: Optional[str]
    text_field: Optional[str]
    # add other expected fields...

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(data: SingleRecord):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict()])
    return predict_df(df)

def predict_df(df: pd.DataFrame):
    # 1) Preprocess using the same feature_engineering
    X = feature_engineering(df.copy())

    # 2) Align columns to training features (fill missing with 0)
    if FEATURE_COLUMNS is not None:
        for c in FEATURE_COLUMNS:
            if c not in X.columns:
                X[c] = 0
        # keep same column order
        X = X[FEATURE_COLUMNS]
    else:
        # If no saved feature list, sort columns to ensure deterministic order
        X = X.reindex(sorted(X.columns), axis=1)

    # 3) Predict
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = [{"probability": float(p), "pred": int(pred)} for p, pred in zip(probs, preds)]
    else:
        preds = MODEL.predict(X)
        out = [{"pred": int(p)} for p in preds]

    return {"predictions": out, "n": len(out)}