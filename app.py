from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.spam_email.data import load_dataset
from src.spam_email.model import train, save_model, load_model, predict_text
from src.spam_email.visualize import plot_confusion_matrix, plot_roc_curve


st.set_page_config(page_title="Spam Email Classifier", layout="wide")

st.title("Spam Email Classifier")
st.caption("End-to-end pipeline: preprocessing, training, metrics, and live prediction")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV with label,text columns", type=["csv"])  # optional
    default_path = st.text_input("Or use dataset path", value="data/sample_spam.csv")
    model_choice = st.selectbox("Model", options=["logreg", "nb"], index=0)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    train_btn = st.button("Train Model")


def read_df(uploaded_file, default_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_dataset(default_path)
    return df


if "train_result" not in st.session_state:
    st.session_state.train_result = None

df = read_df(uploaded, default_path)

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

if train_btn:
    with st.spinner("Training model..."):
        res = train(df, model=model_choice, test_size=float(test_size))
        st.session_state.train_result = res
    st.success("Training complete")


res = st.session_state.train_result
if res is not None:
    st.subheader("Metrics")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{res.metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{res.metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{res.metrics['recall']:.3f}")
    cols[3].metric("F1", f"{res.metrics['f1']:.3f}")
    cols[4].metric("ROC AUC", f"{res.metrics['roc_auc']:.3f}" if np.isfinite(res.metrics['roc_auc']) else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_confusion_matrix(res.confusion_matrix))
    with c2:
        if res.roc is not None:
            fpr, tpr, _ = res.roc
            st.pyplot(plot_roc_curve(fpr, tpr, res.metrics["roc_auc"]))
        else:
            st.info("ROC curve unavailable for this model.")

    st.subheader("Classification Report")
    st.code(res.classification_report)

    st.subheader("Try a message")
    user_text = st.text_area("Enter email/message text", height=120)
    if st.button("Predict") and user_text.strip():
        out = predict_text(res.pipeline, user_text)
        label = "spam" if out["pred"] == 1 else "ham"
        st.write(f"Prediction: :blue[{label}]  |  P(spam) = {out['proba_spam']}")

    with st.expander("Export trained model"):
        model_path = st.text_input("Model output path", value="artifacts/model.joblib")
        if st.button("Save model"):
            os.makedirs(Path(model_path).parent, exist_ok=True)
            save_model(res.pipeline, model_path)
            st.success(f"Saved to {model_path}")

