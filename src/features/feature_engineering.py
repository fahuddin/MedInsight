import pandas as pd

def feature_engineering(df):
    df = df.drop(columns=["patient_id", "clinical_notes"], errors="ignore")
    
    # One-hot encode gender and smoker
    df = pd.get_dummies(df, columns=["gender", "smoker"], drop_first=True)
    
    return df
