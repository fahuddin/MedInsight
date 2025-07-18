# src/models/train_model.py

import pandas as pd
import joblib
from xgboost import XGBClassifier
from features.feature_engineering import feature_engineering

def train_and_save_model(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    df = df.dropna()

    X = feature_engineering(df)
    y = df["readmission"]

    model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
