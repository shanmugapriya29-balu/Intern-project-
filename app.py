import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

NUM_COLS = ["Age", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]

FEATURE_LABELS = {
    "Age": "Age (years)",
    "ALB": "Albumin (ALB, g/L)",
    "ALP": "Alkaline Phosphatase (ALP, U/L)",
    "ALT": "Alanine Aminotransferase (ALT, U/L)",
    "AST": "Aspartate Aminotransferase (AST, U/L)",
    "BIL": "Bilirubin (BIL, µmol/L)",
    "CHE": "Cholinesterase (CHE, U/L)",
    "CHOL": "Cholesterol (CHOL, mmol/L)",
    "CREA": "Creatinine (CREA, µmol/L)",
    "GGT": "Gamma-Glutamyl Transferase (GGT, U/L)",
    "PROT": "Total Protein (PROT, g/L)",
}

CLASS_LABELS = {
    "0=Blood Donor": "Healthy (Blood Donor)",
    "1=Hepatitis": "Hepatitis",
    "2=Fibrosis": "Fibrosis",
    "3=Cirrhosis": "Cirrhosis",
}

st.set_page_config(
    page_title="Hepatitis C Stage Predictor",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_meta():
    with open("metrics.json") as f:
        metrics = json.load(f)
    with open("feature_ranges.json") as f:
        ranges = json.load(f)
    with open("class_counts.json") as f:
        class_counts = json.load(f)
    return metrics, ranges, class_counts


@st.cache_data
def load_data():
    df = pd.read_csv("HepatitisCdata.csv", index_col=0)
    df["Category"] = df["Category"].str.replace("0s=suspect Blood Donor", "0=Blood Donor")
    return df


model = load_model()
metrics, ranges, class_counts = load_meta()
df = load_data()

st.title("🩺 Hepatitis C Stage Predictor")
st.write(
    "Predicts likely diagnostic category (healthy blood donor, hepatitis, "
    "fibrosis, or cirrhosis) from routine blood test values, using a "
    "Logistic Regression model trained on the UCI Hepatitis C dataset."
)

st.warning(
    "⚠️ **This is an educational demo, not a medical device.** "
    "Predictions are based on a small dataset (615 patients) and must never "
    "be used for actual diagnosis or treatment decisions. Always consult a "
    "qualified healthcare professional.",
    icon="⚠️",
)

with st.sidebar:
    st.header("About this model")
    st.metric("Accuracy (test set)", metrics["accuracy"])
    st.metric("Balanced Accuracy", metrics["balanced_accuracy"])
    st.metric("Macro F1", metrics["f1_macro"])
    st.caption(
        f"Trained on {metrics['n_train']} records, evaluated on "
        f"{metrics['n_test']} held-out records."
    )
    st.divider()
    st.write("**Class distribution in data:**")
    for cls, count in class_counts.items():
        st.caption(f"{CLASS_LABELS.get(cls, cls)}: {count}")
    st.divider()
    st.caption(
        "Model: Logistic Regression with balanced class weights "
        "(median imputation + standard scaling + one-hot encoding for sex). "
        "Chosen over Random Forest and Gradient Boosting because it gave far "
        "better balanced accuracy / macro-F1 on the minority disease classes, "
        "which matters more than raw accuracy on this imbalanced dataset."
    )

st.subheader("Enter patient blood test values")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        FEATURE_LABELS["Age"],
        min_value=int(ranges["Age"]["min"]),
        max_value=int(ranges["Age"]["max"]),
        value=round(ranges["Age"]["median"]),
    )
    sex = st.radio("Sex", options=["m", "f"], horizontal=True, format_func=lambda x: "Male" if x == "m" else "Female")
    alb = st.number_input(FEATURE_LABELS["ALB"], value=round(ranges["ALB"]["median"], 1), step=0.1, format="%.1f")
    alp = st.number_input(FEATURE_LABELS["ALP"], value=round(ranges["ALP"]["median"], 1), step=0.1, format="%.1f")
    alt = st.number_input(FEATURE_LABELS["ALT"], value=round(ranges["ALT"]["median"], 1), step=0.1, format="%.1f")
    ast = st.number_input(FEATURE_LABELS["AST"], value=round(ranges["AST"]["median"], 1), step=0.1, format="%.1f")

with col2:
    bil = st.number_input(FEATURE_LABELS["BIL"], value=round(ranges["BIL"]["median"], 1), step=0.1, format="%.1f")
    che = st.number_input(FEATURE_LABELS["CHE"], value=round(ranges["CHE"]["median"], 2), step=0.01, format="%.2f")
    chol = st.number_input(FEATURE_LABELS["CHOL"], value=round(ranges["CHOL"]["median"], 2), step=0.01, format="%.2f")
    crea = st.number_input(FEATURE_LABELS["CREA"], value=round(ranges["CREA"]["median"], 1), step=0.1, format="%.1f")
    ggt = st.number_input(FEATURE_LABELS["GGT"], value=round(ranges["GGT"]["median"], 1), step=0.1, format="%.1f")
    prot = st.number_input(FEATURE_LABELS["PROT"], value=round(ranges["PROT"]["median"], 1), step=0.1, format="%.1f")

st.divider()

if st.button("Predict Category", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[age, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot, sex]],
        columns=NUM_COLS + ["Sex"],
    )

    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps["clf"].classes_
    pred_class = classes[np.argmax(proba)]

    label = CLASS_LABELS.get(pred_class, pred_class)
    if pred_class == "0=Blood Donor":
        st.success(f"### Predicted: {label}")
    else:
        st.error(f"### Predicted: {label}")

    proba_df = pd.DataFrame(
        {"Category": [CLASS_LABELS.get(c, c) for c in classes], "Probability": proba}
    ).sort_values("Probability", ascending=False)

    st.write("**Prediction probabilities:**")
    st.dataframe(
        proba_df.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(proba_df.set_index("Category")["Probability"])

st.divider()

with st.expander("📊 Explore the training data"):
    st.write(f"Dataset contains **{len(df)}** patient records.")
    st.write("Category counts:")
    st.bar_chart(df["Category"].value_counts())

    st.write("Summary statistics of blood markers:")
    st.dataframe(df[NUM_COLS].describe().T, use_container_width=True)

    st.write("Average marker values by category:")
    st.dataframe(df.groupby("Category")[NUM_COLS].mean().round(2), use_container_width=True)

with st.expander("ℹ️ How the model works"):
    st.write(
        "The model is a **Logistic Regression** classifier (one-vs-rest, "
        "balanced class weights) trained on 11 blood markers plus age and sex. "
        "Missing values are imputed with the column median, and numeric "
        "features are standardized before fitting."
    )
    st.write("Full test-set classification report:")
    report_df = pd.DataFrame(metrics["per_class_report"]).T
    report_df.index = [CLASS_LABELS.get(i, i) for i in report_df.index]
    st.dataframe(report_df.round(3), use_container_width=True)
