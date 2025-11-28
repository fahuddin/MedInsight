# src/models/train_model.py


import os
import sys
import joblib
import pandas as pd
from typing import Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,f1_score
)

# When running this file directly (e.g. `python train_model.py` from
# `src/models`), the package `src` may not be on sys.path. Add the
# repository root to `sys.path` so `from src.features...` works.
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.features.feature_engineering import feature_engineering

DEFAULT_DATA_PATH = os.path.join(
    "C:\\Users\\fahud\\medinsight", 
    "data", 
    "Synthetic_Clinical_Data.csv"
)
DEFAULT_MODEL_PATH = "models/xgb_readmission_model.pkl"
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
    print(df.head())
    df = df.dropna(subset=["readmission"])  # ensure target present

    print("Label distribution (full):")
    print(df["readmission"].value_counts(normalize=True))


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
    print("pos:", pos, "neg:", neg, "scale_pos_weight:", spw)
      # 5) Define model (XGBoost 2.x style)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    
    

    params = { "objective": "binary:logistic", 
              "eval_metric": "logloss", 
              "eta": 0.1, # a bit faster learning 
              "max_depth": 3, # simpler trees 
              "subsample": 0.8, # use 80% of rows per tree 
              "colsample_bytree": 0.8, # use 80% of features per tree 
              "tree_method": "hist", 
              "scale_pos_weight": spw, # you can keep this for now 
              "seed": RANDOM_STATE, }

    evals = [(dtrain, "train"), (dval, "validation")]

    # 8) Train with early stopping
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=True,
    )



    best_iter = getattr(booster, "best_iteration", None)
    print("Best iteration:", best_iter)

    y_val_proba = booster.predict(dval)

    best_t = 0.5
    best_f1 = -1

    for t in [i/100 for i in range(5, 95, 5)]:
        preds = (y_val_proba >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("Best threshold:", best_t, "Best F1:", best_f1)

    # 9) Evaluate on TEST
    y_proba = booster.predict(dtest)  # probabilities for class 1

    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print("=== Test Metrics ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Average Precision (AUPRC): {ap:.4f}")

    # Use the best threshold from validation:
    y_pred = (y_proba >= best_t).astype(int)
    print(f"\nClassification Report @ threshold={best_t:.2f}:")
    print(classification_report(y_test, y_pred, digits=4))

    # 10) Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(booster, model_path)
    print(f"✅ Booster saved to {model_path} (best_iteration={best_iter})")

    return roc, ap


if __name__ == "__main__":
    train_and_save_model(DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH)
