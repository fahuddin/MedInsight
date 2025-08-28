import pandas as pd
import numpy as np
from nlp.text_cleaning import clean_text,lemmatize
from nlp.ner_extraction import extract_entities
from nlp.embeddings import embed_text


def feature_engineering(df):
    df = df.drop(columns=["patient_id"], errors="ignore")

    notes = df["clinical_notes"].fillna("").apply(clean_text).apply(lemmatize).tolist()

    structured = df.drop(columns=["clinical_notes","readmission"], errors="ignore")

    structured = pd.get_dummies(structured, columns=["gender", "smoker"], drop_first=True)

    X_struct = structured.to_numpy()

    X_notes = embed_text(notes).numpy()

    X = np.hstack([X_struct, X_notes])
    # One-hot encode gender and smoker
    
    return df
