"""
Trains a Logistic Regression classifier on HepatitisCdata.csv to predict
disease category from blood test values, and saves:
  - model.joblib          (full sklearn pipeline: imputer + scaler + one-hot + logistic regression)
  - metrics.json           (accuracy, balanced accuracy, macro F1, per-class report)
  - feature_ranges.json    (min/max/mean of each numeric feature, for input widget bounds)
  - class_counts.json      (class distribution, for display)

Run this once (`python train_model.py`) before launching the Streamlit app.
The dataset (HepatitisCdata.csv) must be in the same folder.
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

DATA_PATH = "HepatitisCdata.csv"
NUM_COLS = ["Age", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
CAT_COLS = ["Sex"]
TARGET = "Category"


def main():
    df = pd.read_csv(DATA_PATH, index_col=0)

    # Merge the "suspect blood donor" label into the main blood donor class —
    # it's a borderline/uncertain label for the same underlying class (healthy),
    # and keeping it separate would create a 5th class with almost no data.
    df[TARGET] = df[TARGET].str.replace("0s=suspect Blood Donor", "0=Blood Donor")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM_COLS),
        ("cat", OneHotEncoder(drop="if_binary"), CAT_COLS),
    ])

    # class_weight='balanced' matters a lot here: ~88% of rows are the
    # healthy "Blood Donor" class, and the three disease classes
    # (Hepatitis, Fibrosis, Cirrhosis) have only 20-30 examples each.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", clf),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": list(pipeline.named_steps["clf"].classes_),
        "per_class_report": report,
    }

    ranges = {}
    for col in NUM_COLS:
        ranges[col] = {
            "min": float(np.nanmin(df[col])),
            "max": float(np.nanmax(df[col])),
            "mean": float(np.nanmean(df[col])),
            "median": float(np.nanmedian(df[col])),
        }

    class_counts = df[TARGET].value_counts().to_dict()

    joblib.dump(pipeline, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("feature_ranges.json", "w") as f:
        json.dump(ranges, f, indent=2)
    with open("class_counts.json", "w") as f:
        json.dump(class_counts, f, indent=2)

    print("Model trained and saved.")
    print("Accuracy:", metrics["accuracy"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Macro F1:", metrics["f1_macro"])


if __name__ == "__main__":
    main()
