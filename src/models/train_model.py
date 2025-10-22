# src/models/train_model.py

import os
import joblib
import pandas as pd
from typing import Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from src.features.feature_engineering import feature_engineering

DEFAULT_MODEL_PATH = "models/xgb_readmission_model.pkl"
DEFAULT_DATA_PATH = "./data/Synthetic_Clinical_Data.csv"
RANDOM_STATE = 42

def train_and_save_model(data_path: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Train XGBoost on a train split, validate with early stopping, and evaluate on a held-out test split.
    Saves the best model to disk.

    Returns:
        (roc_auc, average_precision) on the test set.
    """
    # 1) Load
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["readmission"])  # ensure target present

    # 2) Features / target
    X_full = feature_engineering(df)
    y_full = df["readmission"].astype(int)

    # 3) Train / test split (stratify keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )

    # 4) Further split train into train/valid for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    # Optional: handle imbalance (set scale_pos_weight ≈ neg/pos ratio in TRAIN)
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = max(1.0, neg / max(1, pos))  # avoid zero

    # 5) Model
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",  # fast CPU
        # remove use_label_encoder (deprecated in recent xgboost)
    )

    # 6) Train with early stopping on validation
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",           # use AUC for early stopping
        verbose=False,
        early_stopping_rounds=50
    )

    # 7) Evaluate on TEST (never seen during training)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision function or predictions
        y_proba = model.predict(X_test)

    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print("=== Test Metrics ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Average Precision (AUPRC): {ap:.4f}")

    # Also show classification report at a default threshold (0.5)
    y_pred = (y_proba >= 0.5).astype(int)
    print("\nClassification Report @ 0.5:")
    print(classification_report(y_test, y_pred, digits=4))

    # 8) Save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path} (best_iteration={model.get_booster().best_iteration})")
