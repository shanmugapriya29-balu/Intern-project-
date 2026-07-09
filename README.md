# Hepatitis C Stage Predictor

A Streamlit web app that predicts a patient's Hepatitis C diagnostic
category (healthy blood donor, hepatitis, fibrosis, or cirrhosis) from
routine blood test values, using a Logistic Regression model trained on
`HepatitisCdata.csv`.

**⚠️ Educational demo only — not a medical device. Do not use for real
diagnosis.**

## Files

| File | Purpose |
|---|---|
| `HepatitisCdata.csv` | Training data (615 patient records) |
| `train_model.py` | Trains the model and saves `model.joblib`, `metrics.json`, `feature_ranges.json`, `class_counts.json` |
| `app.py` | The Streamlit web application |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model (run once, or whenever the CSV changes)

```bash
python train_model.py
```

This reads `HepatitisCdata.csv` and produces:
- `model.joblib` — fitted scikit-learn pipeline (median imputer + scaler + one-hot encoder + logistic regression)
- `metrics.json` — accuracy, balanced accuracy, macro F1, full per-class report on a held-out test set
- `feature_ranges.json` — min/max/mean/median of each numeric feature (used for input widget bounds)
- `class_counts.json` — class distribution

## 2. Run the app

```bash
streamlit run app.py
```

Opens the app in your browser (default: http://localhost:8501).

## Dataset & problem

The target column `Category` has 5 raw values:
- `0=Blood Donor` / `0s=suspect Blood Donor` — merged into one "healthy" class during training (the "suspect" label had only 7 rows)
- `1=Hepatitis`
- `2=Fibrosis`
- `3=Cirrhosis`

The dataset is heavily imbalanced: ~88% of patients are healthy blood
donors, with only 20–30 examples of each disease class.

Features: Age, Sex, and 11 blood markers (ALB, ALP, ALT, AST, BIL, CHE,
CHOL, CREA, GGT, PROT). A few markers have missing values, imputed with
the column median.

## Model selection

Three models were compared on a stratified 80/20 split:

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| **Logistic Regression (balanced)** | **0.951** | **0.851** | **0.790** |
| Gradient Boosting | 0.943 | 0.689 | 0.708 |
| Random Forest (balanced) | 0.935 | 0.592 | 0.641 |

Logistic Regression with `class_weight='balanced'` was chosen because it
performs far better on **balanced accuracy** and **macro F1** — the metrics
that matter most given the severe class imbalance. Plain accuracy is
misleading here since a model that always predicts "Blood Donor" would
already score ~88% accuracy while being clinically useless. 5-fold
cross-validated balanced accuracy for the final model: ~0.67 (varies
across folds due to the very small disease-class sample sizes).

## Retraining on new data

Replace `HepatitisCdata.csv` with a new file using the same column names
and category format, then re-run `python train_model.py`.
