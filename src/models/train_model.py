# src/models/train_model.py

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from features.feature_engineering import feature_engineering

def train_and_save_model(data_path: str):
    df = pd.read_csv(data_path)
    df = df.dropna()

    X = feature_engineering(df)
    y = df["readmission"]

    model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    model_path = "models/xgb_readmission_model.pkl"

    os.makedir(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    
