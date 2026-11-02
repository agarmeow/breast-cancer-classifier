"""
Web app for the Breast Cancer Prediction project.
Local:   python app.py   then open http://127.0.0.1:5000  (run `python train_model.py` once first)
Render:  gunicorn app:app
Vercel:  detected automatically (this file must define `app`)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "meta.json")

if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    raise SystemExit("Model not found. Run `python train_model.py` first.")

model = joblib.load(MODEL_PATH)
with open(META_PATH) as fh:
    META = json.load(fh)

FEATURES = [f["name"] for f in META["features"]]
RANGES = {f["name"]: f for f in META["features"]}
scaler = model.named_steps["sc"]
clf = model.named_steps["m"]

app = Flask(__name__, static_folder="public", static_url_path="")


def explain(x_row):
    """Per-feature push towards 'malignant' (positive) or 'benign' (negative).
    For logistic regression: contribution = coefficient x standardised value."""
    z = scaler.transform(x_row)[0]
    contrib = clf.coef_[0] * z
    order = np.argsort(-np.abs(contrib))[:3]
    return [{"feature": FEATURES[i],
             "direction": "malignant" if contrib[i] > 0 else "benign"} for i in order]


DISPLAY_ORDER = ["radius", "perimeter", "area", "concavity", "concave points"]


def display_meta():
    """Same data, but with fields in a natural reading order for the form."""
    def key(f):
        base = f["name"].replace("mean ", "").replace("worst ", "")
        return DISPLAY_ORDER.index(base)
    shown = dict(META)
    shown["features"] = sorted(META["features"], key=key)
    return shown


@app.route("/")
def index():
    return render_template("index.html", meta=display_meta())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    values, errors, warnings = [], {}, []

    for name in FEATURES:
        raw = data.get(name)
        try:
            v = float(raw)
            if not np.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errors[name] = "Enter a number"
            continue
        if v < 0 or (v == 0 and RANGES[name]["min"] > 0):
            errors[name] = "Cannot be negative" if v < 0 else "Must be greater than 0"
            continue
        lo, hi = RANGES[name]["min"], RANGES[name]["max"]
        if v < lo * 0.5 or v > hi * 1.5:
            errors[name] = "Far outside the range this model was trained on (%g to %g)" % (lo, hi)
            continue
        if v < lo or v > hi:
            warnings.append(name)
        values.append(v)

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    x = pd.DataFrame([values], columns=FEATURES)
    p_malignant = float(model.predict_proba(x)[0][1])
    label = "Malignant" if p_malignant >= 0.5 else "Benign"
    return jsonify({
        "ok": True,
        "label": label,
        "p_malignant": round(p_malignant, 4),
        "p_benign": round(1 - p_malignant, 4),
        "drivers": explain(x),
        "out_of_range": warnings,
    })


if __name__ == "__main__":
    app.run(debug=False)
