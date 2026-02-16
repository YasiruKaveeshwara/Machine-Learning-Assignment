from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(
    fig, out_path: Optional[Path] = None, dpi: int = 150, close: bool = True
) -> None:
    if out_path is None:
        return
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    out_path: Optional[Path] = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, values_format="d"
    )
    ax.set_title(title)
    save_fig(fig, out_path, dpi=dpi, close=True)
    return fig, ax, disp


def plot_roc_curve(
    y_true,
    y_proba,
    title: str = "ROC Curve",
    out_path: Optional[Path] = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    save_fig(fig, out_path, dpi=dpi, close=True)
    return fig, ax, disp


def plot_pr_curve(
    y_true,
    y_proba,
    title: str = "Precision-Recall Curve",
    out_path: Optional[Path] = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    save_fig(fig, out_path, dpi=dpi, close=True)
    return fig, ax, disp


def plot_feature_importance(
    model,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance (Top)",
    out_path: Optional[Path] = None,
    dpi: int = 150,
):
    """
    Works for tree models with feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_ attribute.")

    importances = np.asarray(model.feature_importances_)
    idx = np.argsort(importances)[::-1][:top_n]

    labels = (
        [f"f{i}" for i in idx]
        if feature_names is None
        else [feature_names[i] for i in idx]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(idx))[::-1], importances[idx])
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels(labels)  # type: ignore
    ax.set_title(title)
    ax.set_xlabel("Importance")
    save_fig(fig, out_path, dpi=dpi, close=True)
    return fig, ax
