from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocess import make_sklearn_preprocessor


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    roc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    classification_report: str


def build_pipeline(model: str = "logreg", max_features: int = 20000) -> Pipeline:
    preprocessor = make_sklearn_preprocessor()
    tfidf = TfidfVectorizer(
        preprocessor=preprocessor,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=None) if model == "logreg" else MultinomialNB()

    pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )
    return pipeline


def train(df: pd.DataFrame, model: str = "logreg", test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    X = df["text"].astype(str)
    y = (df["label"].str.lower() == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(model=model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # Some models may not support predict_proba
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0)

    roc = None
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc = (fpr, tpr, thresholds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }

    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        confusion_matrix=cm,
        roc=roc,
        classification_report=report,
    )


def save_model(pipeline: Pipeline, path: str) -> None:
    dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return load(path)


def predict_text(pipeline: Pipeline, text: str) -> Dict[str, float | int]:
    proba = None
    try:
        proba = float(pipeline.predict_proba([text])[0, 1])
    except Exception:
        proba = None
    pred = int(pipeline.predict([text])[0])
    return {"pred": pred, "proba_spam": proba}

