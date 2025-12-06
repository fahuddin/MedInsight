# src/models/evaluate_model.py

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.features.feature_engineering import feature_engineering

def evaluate(model_path: str, test_data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    df = df.dropna()

    X = feature_engineering(df)
    y = df["readmission"]

    y_pred = model.predict(X)

    print("Classification Report:")
    print(classification_report(y, y_pred))

    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
