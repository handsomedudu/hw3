from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from src.spam_email.data import load_dataset
from src.spam_email.model import train, save_model, load_model, predict_text


app = typer.Typer(help="Spam Email ML CLI")


@app.command()
def train_cmd(
    data_path: str = typer.Option("data/sample_spam.csv", help="Path to CSV with label,text columns"),
    model_out: str = typer.Option("artifacts/model.joblib", help="Where to save trained model"),
    model: str = typer.Option("logreg", help="Classifier: logreg|nb"),
):
    df = load_dataset(data_path)
    res = train(df, model=model)
    os.makedirs(Path(model_out).parent, exist_ok=True)
    save_model(res.pipeline, model_out)
    typer.echo("Training complete. Metrics:")
    for k, v in res.metrics.items():
        typer.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    typer.echo("\nClassification Report:\n" + res.classification_report)


@app.command()
def evaluate(
    data_path: str = typer.Option("data/sample_spam.csv", help="Path to CSV with label,text columns"),
    model_path: str = typer.Option("artifacts/model.joblib", help="Trained model path"),
):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import pandas as pd

    df = load_dataset(data_path)
    pipe = load_model(model_path)
    X = df["text"].astype(str)
    y = (df["label"].str.lower() == "spam").astype(int)
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    typer.echo(f"accuracy={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")


@app.command()
def predict(
    model_path: str = typer.Option("artifacts/model.joblib", help="Trained model path"),
    text: str = typer.Option(..., help="Email/message text to classify"),
):
    pipe = load_model(model_path)
    out = predict_text(pipe, text)
    label = "spam" if out["pred"] == 1 else "ham"
    typer.echo(f"label={label} proba_spam={out['proba_spam']}")


if __name__ == "__main__":
    app()

