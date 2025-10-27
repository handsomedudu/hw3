from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.spam_email.data import load_dataset
from src.spam_email.model import train, save_model, predict_text
from src.spam_email.visualize import plot_confusion_matrix, plot_roc_curve

st.set_page_config(page_title="Spam Email Classifier", layout="wide")

st.title("Spam Email Classifier")
st.caption("EDA, training, metrics, PR/ROC, and live prediction")

with st.sidebar:
    st.header("資料與設定")
    uploaded = st.file_uploader("上傳 CSV（需含 label,text 欄位）", type=["csv"])  # optional
    default_path = st.text_input("或使用路徑", value="data/sample_spam.csv")
    st.divider()
    st.subheader("向量化設定")
    model_choice = st.selectbox("模型", options=["logreg", "nb"], index=0)
    max_features = st.slider("max_features", 2000, 50000, 20000, 1000)
    ngram = st.select_slider("ngram_range", options=["1", "1-2"], value="1-2")
    test_size = st.slider("測試集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    train_btn = st.button("開始訓練")


def read_df(uploaded_file, default_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_dataset(default_path)
    return df


# Helpers to align with reference demo
import re
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")


def normalize_text(text: str, keep_numbers: bool = False) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    if not keep_numbers:
        t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


from collections import Counter


def token_topn(series: pd.Series, topn: int) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)


def list_datasets() -> list[str]:
    paths: list[str] = []
    for root in ("datasets", os.path.join("datasets", "processed")):
        if os.path.isdir(root):
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if name.lower().endswith(".csv") and os.path.isfile(p):
                    paths.append(p)
    if not paths and os.path.exists("data/sample_spam.csv"):
        paths.append("data/sample_spam.csv")
    return sorted(paths)


if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None

df = read_df(uploaded, default_path)
st.session_state.last_df = df

tab_data, tab_eda, tab_train, tab_metrics, tab_predict = st.tabs([
    "資料", "EDA", "訓練", "指標/視覺化", "即時預測"
])

with tab_data:
    st.subheader("資料預覽")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"筆數：{len(df)} | 欄位：{', '.join(df.columns)}")

with tab_eda:
    st.subheader("標籤分佈")
    counts = df["label"].value_counts()
    st.bar_chart(counts)

    st.subheader("訊息長度分佈")
    lens = df["text"].astype(str).str.len()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lens, bins=30, color="#4C78A8")
    ax.set_xlabel("Length (chars)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab_train:
    st.subheader("模型訓練")
    st.caption("選好左側參數後點擊按鈕開始訓練")
    if train_btn:
        with st.spinner("Training model..."):
            from src.spam_email import model as model_mod
            orig_build = model_mod.build_pipeline

            def custom_build(model: str = "logreg", max_features_param: int = 20000):
                pipe = orig_build(model=model, max_features=max_features)
                if ngram == "1":
                    pipe.named_steps["tfidf"].ngram_range = (1, 1)
                else:
                    pipe.named_steps["tfidf"].ngram_range = (1, 2)
                return pipe

            model_mod.build_pipeline = custom_build
            try:
                res = train(df, model=model_choice, test_size=float(test_size))
            finally:
                model_mod.build_pipeline = orig_build
            st.session_state.train_result = res
        st.success("訓練完成")
    else:
        st.info("尚未開始訓練")

res = st.session_state.train_result
with tab_metrics:
    st.subheader("整體指標")
    if res is None:
        st.info("請先前往『訓練』分頁進行模型訓練")
    else:
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
                st.info("此模型無 ROC 曲線")

        st.subheader("分類報告")
        st.code(res.classification_report)

        from src.spam_email.model import top_features, misclassified_examples
        st.subheader("重要特徵 Top-K")
        feats = top_features(res.pipeline, k=20)
        if feats is None:
            st.info("無法取得特徵重要性（模型不支援或特徵不可用）")
        else:
            c3, c4 = st.columns(2)
            with c3:
                st.write("Top Spam Features")
                st.table(feats["spam"])  # list of (feature, weight)
            with c4:
                st.write("Top Ham Features")
                st.table(feats["ham"])  # list of (feature, weight)

        st.subheader("錯誤案例（前10筆）")
        y_series = (df["label"].str.lower() == "spam").astype(int)
        wrong = misclassified_examples(res.pipeline, df["text"].astype(str), y_series, n=10)
        if wrong.empty:
            st.success("目前沒有錯誤分類的案例（或樣本太少）")
        else:
            wrong = wrong.assign(true_label=lambda d: d.true.map({0: "ham", 1: "spam"}),
                                 pred_label=lambda d: d.pred.map({0: "ham", 1: "spam"}))
            st.dataframe(wrong[["text", "true_label", "pred_label"]], use_container_width=True)

with tab_predict:
    st.subheader("即時預測與模型檔案")
    if res is None:
        st.info("請先完成模型訓練，或在專案根目錄放置 models/ 以載入既有模型")
    else:
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1:
            if st.button("套用 Spam 範例"):
                st.session_state["input_text"] = "Free entry in 2 a wkly comp to win cash now! Call +44 906-170-1461 to claim prize"
        with c_ex2:
            if st.button("套用 Ham 範例"):
                st.session_state["input_text"] = "Ok, I'll see you at 7 pm for dinner. Thanks!"

        if "input_text" not in st.session_state:
            st.session_state["input_text"] = ""

        user_text = st.text_area("輸入要判斷的訊息文字", key="input_text", height=120)
        if st.button("預測") and user_text.strip():
            out = predict_text(res.pipeline, user_text)
            label = "spam" if out["pred"] == 1 else "ham"
            st.write(f"Prediction: :blue[{label}]  |  P(spam) = {out['proba_spam']}")

        with st.expander("匯出模型"):
            model_path = st.text_input("輸出路徑", value="artifacts/model.joblib")
            if st.button("儲存模型"):
                os.makedirs(Path(model_path).parent, exist_ok=True)
                save_model(res.pipeline, model_path)
                st.success(f"Saved to {model_path}")

    # Additional: Use pre-trained artifacts in models/ to mimic demo visuals
    st.divider()
    st.subheader("使用 models/ 既有模型的可視化")
    models_dir = st.text_input("Models 目錄", value="models")
    threshold = st.slider("決策門檻", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
    try:
        import joblib
        from sklearn.metrics import precision_recall_curve, auc as sk_auc, roc_curve, confusion_matrix
        from sklearn.model_selection import train_test_split

        vec_p = os.path.join(models_dir, "spam_tfidf_vectorizer.joblib")
        clf_p = os.path.join(models_dir, "spam_logreg_model.joblib")
        if os.path.exists(vec_p) and os.path.exists(clf_p):
            vec = joblib.load(vec_p)
            clf = joblib.load(clf_p)
            X = df["text"].astype(str).fillna("")
            y = (df["label"].astype(str).str.lower() == "spam").astype(int).values
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=42, stratify=y)
            Xte_vec = vec.transform(Xte)
            proba = clf.predict_proba(Xte_vec)[:, 1]
            pred = (proba >= threshold).astype(int)

            cm = confusion_matrix(yte, pred)
            cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]) 
            st.write("Confusion matrix (models/)")
            st.dataframe(cm_df)

            fpr, tpr, _ = roc_curve(yte, proba)
            roc_auc = sk_auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(yte, proba)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0].set_title("ROC")
            ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
            from sklearn.metrics import PrecisionRecallDisplay
            PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax[1])
            ax[1].set_title("Precision-Recall")
            st.pyplot(fig)

            st.write("Threshold sweep (precision/recall/f1)")
            ths = np.round(np.linspace(0.3, 0.8, 11), 3)
            rows = []
            from sklearn.metrics import precision_score, recall_score, f1_score
            for t in ths:
                p = (proba >= t).astype(int)
                rows.append({
                    "threshold": t,
                    "precision": float(precision_score(yte, p, zero_division=0)),
                    "recall": float(recall_score(yte, p, zero_division=0)),
                    "f1": float(f1_score(yte, p, zero_division=0)),
                })
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("未找到 models/ 下的預訓練模型檔案，若要呈現與 Demo 相同的圖表，請放置 spam_tfidf_vectorizer.joblib 與 spam_logreg_model.joblib。")
    except Exception as e:
        st.warning(f"無法載入 models/ 模型或繪圖：{e}")

