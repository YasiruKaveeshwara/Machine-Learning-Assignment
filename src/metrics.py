from __future__ import annotations

from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics in a JSON-serializable format.
    y_proba should be probability of positive class (shape: [n_samples]).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),  # JSON-safe
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba).reshape(-1)

        # ROC-AUC requires both classes present
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")

        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            out["pr_auc"] = float("nan")

        # log_loss requires valid probs
        try:
            out["log_loss"] = float(
                log_loss(y_true, np.vstack([1 - y_proba, y_proba]).T)
            )
        except ValueError:
            out["log_loss"] = float("nan")

    return out


def metrics_table(
    rows: List[Dict[str, Any]], model_name_key: str = "model"
) -> pd.DataFrame:
    """
    Convert list of metric dicts into a clean DataFrame.
    Removes confusion_matrix for table display.
    """
    df = pd.DataFrame(rows).copy()
    if "confusion_matrix" in df.columns:
        df = df.drop(columns=["confusion_matrix"])

    preferred = [
        model_name_key,
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]


def format_metrics_for_print(metrics: Dict[str, Any]) -> str:
    """
    Human-readable metrics string for notebook/script logs.
    """
    keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
    ]
    parts = []
    for k in keys:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
    return " | ".join(parts)
