"""
Trains the final model and saves it for the web app.
Run once:  python train_model.py
Creates model/model.joblib and model/meta.json
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src import data_loader, features, models
from src.config import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Short plain-language hints shown under each input in the web app
HINTS = {
    "radius": "Average distance from the centre of the nucleus to its edge",
    "perimeter": "Length of the boundary around the nucleus",
    "area": "Area covered by the nucleus",
    "concavity": "Severity of the inward dents in the outline",
    "concave points": "Number of inward dents in the outline",
}


def hint_for(name):
    for key, text in HINTS.items():
        if name.endswith(key):
            return text
    return ""


def main():
    df = data_loader.load_data()
    selected = features.select_features(df)
    X_train, X_test, y_train, y_test = features.split_data(df, selected)

    model = models.tune_logistic_regression(X_train, y_train)   # final model: tuned Logistic Regression
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "auc": round(roc_auc_score(y_test, prob), 4),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "C": float(model.named_steps["m"].C),
    }

    # Feature ranges (from the whole dataset) used for input validation and warnings
    feats = []
    for f in selected:
        col = df[f]
        feats.append({
            "name": f,
            "group": "worst" if f.startswith("worst") else "mean",
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
            "median": round(float(col.median()), 4),
            "hint": hint_for(f),
        })

    # Two real example patients from the test set (one per class), the ones the model is
    # most typical about, so the demo buttons show clear examples
    samples = {}
    for label, key in ((0, "benign"), (1, "malignant")):
        idx = np.where(y_test.values == label)[0]
        target_p = np.median(prob[idx])
        best = idx[np.argmin(np.abs(prob[idx] - target_p))]
        samples[key] = {f: round(float(X_test.iloc[best][f]), 4) for f in selected}

    meta = {"features": feats, "metrics": metrics, "samples": samples}
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nModel saved to model/model.joblib")
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
