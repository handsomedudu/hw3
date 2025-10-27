from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


def plot_confusion_matrix(cm: np.ndarray, labels=("ham", "spam")):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

