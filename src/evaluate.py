"""Evaluate the trained XGBoost model on the test split.
Reads model.tar.gz from /opt/ml/processing/model and test.csv from
/opt/ml/processing/test, writes evaluation.json (PropertyFile) to
/opt/ml/processing/evaluation.
"""
import os, json, tarfile, pickle
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUT_DIR = "/opt/ml/processing/evaluation"

def main():
    # unpack model.tar.gz
    tp = os.path.join(MODEL_DIR, "model.tar.gz")
    with tarfile.open(tp) as t:
        t.extractall(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "xgboost-model")
    # SageMaker XGBoost 1.x may store the booster as a pickle OR the native binary format.
    # Try native load first, fall back to pickle.
    booster = None
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        print("loaded model via xgb.Booster.load_model")
    except Exception as e1:
        print(f"native load failed ({e1}); trying pickle")
        with open(model_path, "rb") as f:
            booster = pickle.load(f)
        print("loaded model via pickle")

    df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    dmat = xgb.DMatrix(X)
    prob = booster.predict(dmat)
    pred = (prob > 0.5).astype(int)
    acc = float((pred == y).mean())
    tp_ = int(((pred == 1) & (y == 1)).sum()); fp_ = int(((pred == 1) & (y == 0)).sum())
    fn_ = int(((pred == 0) & (y == 1)).sum())
    precision = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
    recall = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0

    report = {"binary_classification_metrics": {
        "accuracy":  {"value": acc,       "standard_deviation": "NaN"},
        "precision": {"value": precision, "standard_deviation": "NaN"},
        "recall":    {"value": recall,    "standard_deviation": "NaN"},
    }}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f)
    print("evaluation:", json.dumps(report))

if __name__ == "__main__":
    main()
