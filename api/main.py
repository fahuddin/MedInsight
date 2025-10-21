from fastapi import FastAPI, UploadFile, File
import pandas as pd

# Import from your modules
from src.features.feature_engineering import feature_engineering
from src.models.train_model import train_and_save_model
from src.models.evaluate import evaluate_model
from src.nlp.text_cleaning import clean_text
from src.nlp.embeddings import generate_embeddings
from src.nlp.ner_extraction import extract_entities


app = FastAPI(title="MediInsight API", version="1.0")

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

async def predict(data: PatientData):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature Engineering
    df_fe = feature_engineering(df)


    return {"predictions": predictions.tolist()}
    