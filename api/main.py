from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import numpy as np
import io
# Import from your modules
from src.features.feature_engineering import feature_engineering
from src.models.train_model import train_and_save_model
from src.models.evaluate import evaluate
from src.nlp.text_cleaning import clean_text
from src.nlp.embeddings import embed_text
from src.nlp.ner_extraction import extract_entities
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import List, Optional
import joblib
import re
import xgboost as xgb

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


def load_artifacts():
    global MODEL, FEATURE_COLUMNS
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURES_PATH):
        FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
    else:
        FEATURE_COLUMNS = None  # fallback: infer runtime but better to save during training
app = FastAPI()
# Register startup handler (do NOT pass load_artifacts as `lifespan=`;
# Starlette may call lifespan callables with an argument. Using the
# startup event calls the function with no args.
app.add_event_handler("startup", load_artifacts)


    # add other expected fields...

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(data: PatientData):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.model_dump()])
    return predict_df(df)

def predict_df(df: pd.DataFrame):
    import numpy as np

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=lambda s: re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).lower())
        if "clinicalnotes" in df.columns and "clinical_notes" not in df.columns:
            df["clinical_notes"] = df.pop("clinicalnotes")
        if "readmission" not in df.columns and "readmit" in df.columns:
            df["readmission"] = df.pop("readmit")
        return df

    df_norm = _normalize_columns(df.copy())

    # 1) Preprocess using the same feature_engineering
    X = feature_engineering(df_norm)  # can be DataFrame or ndarray


    # 3) Predict
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = [{"probability": float(p), "pred": int(pred)} for p, pred in zip(probs, preds)]
    else:
        # Some saved models may be xgboost.Booster and expect a DMatrix
        try:
            preds = MODEL.predict(X)
            out = [{"pred": int(p)} for p in preds]
        except TypeError:
            # MODEL may be an xgboost.Booster requiring a DMatrix.
            # Support both DataFrame and numpy ndarray inputs here.
            data = X.values if hasattr(X, "values") else X
            feature_names = X.columns.tolist() if hasattr(X, "columns") else None
            if feature_names:
                dmat = xgb.DMatrix(data, feature_names=feature_names)
            else:
                dmat = xgb.DMatrix(data)
            probs = MODEL.predict(dmat)
            preds = (probs >= 0.5).astype(int)
            out = [{"probability": float(p), "pred": int(pred)} for p, pred in zip(probs, preds)]

    return {"predictions": out, "n": len(out)}
