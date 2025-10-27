from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DATA_PATH = Path("data") / "sample_spam.csv"


def load_dataset(path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load dataset from CSV.

    Expects at least two columns:
    - label: ham|spam
    - text: message body
    """
    csv_path = Path(path) if path else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Provide a valid --data-path or place a CSV at that location."
        )

    df = pd.read_csv(csv_path)
    required = {"label", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Normalize labels
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df

