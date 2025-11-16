import pandas as pd
import numpy as np
from ..nlp.text_cleaning import clean_text, lemmatize
from ..nlp.embeddings import embed_text



def group_age(series: pd.Series) -> pd.Series:
    """
    Example age bucketing: you can tweak bins as needed.
    """
    return pd.cut(
        series,
        bins=[0, 30, 45, 60, 80, np.inf],
        labels=["<=30", "31-45", "46-60", "61-80", "80+"],
        include_lowest=True,
    )


def feature_engineering(df: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw dataframe and returns a numpy feature matrix X.
    Assumes columns like:
      - 'patient_id' (optional)
      - 'clinical_notes'
      - 'gender'
      - 'smoker'
      - 'age' (optional)
      - 'insurance' (optional)
      - 'readmission' (optional, only for training)
    """

    # Drop ID columns that are just identifiers
    df = df.drop(columns=["patient_id"], errors="ignore")

    # --------- TEXT FEATURES ---------
    # Clean + lemmatize notes, then embed
    notes = (
        df["clinical_notes"]
        .fillna("")
        .apply(clean_text)
        .apply(lemmatize)
        .tolist()
    )

    X_notes = embed_text(notes)
    # in case this returns a torch tensor
    if hasattr(X_notes, "numpy"):
        X_notes = X_notes.numpy()

    # Optional: age bucketing
    if "age" in df.columns:
        df["age_group"] = group_age(df["age"])

    # --------- SIMPLE TEXT STATISTICS (from notes) ---------
    df["note_length_chars"] = df["clinical_notes"].fillna("").str.len()
    df["note_length_words"] = (
        df["clinical_notes"].fillna("").str.split().str.len()
    )

    # --------- STRUCTURED FEATURES MATRIX ---------
    # Drop raw text and label from structured part
    structured = df.drop(
        columns=["clinical_notes", "readmission"],
        errors="ignore",
    )

    # One-hot encode your categoricals
    cat_cols = [c for c in ["gender", "smoker", "age_group"] if c in structured.columns]
    structured = pd.get_dummies(structured, columns=cat_cols, drop_first=True)

    X_struct = structured.to_numpy(dtype=float)

    # Combine structured + note embeddings
    X = np.hstack([X_struct, X_notes])

    return X
