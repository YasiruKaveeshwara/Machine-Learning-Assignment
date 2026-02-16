from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.metrics import compute_classification_metrics


@dataclass(frozen=True)
class TrainOptions:
    random_state: int = 42
    test_size: float = 0.20
    cv_splits: int = 5
    scoring: str = "f1"
    n_jobs: int = -1
    verbose: int = 1


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X (features) and y (target)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series, options: Optional[TrainOptions] = None
):
    """Stratified train/test split."""
    options = options or TrainOptions()
    return train_test_split(
        X,
        y,
        test_size=options.test_size,
        random_state=options.random_state,
        stratify=y,
    )


def get_estimator(model_name: str, random_state: int = 42):
    """Factory for estimators used in the assignment."""
    name = model_name.lower().strip()

    if name in ("logreg", "logistic_regression"):
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name in ("knn", "k_neighbors"):
        return KNeighborsClassifier()
    if name in ("decision_tree", "dt"):
        return DecisionTreeClassifier(random_state=random_state)
    if name in ("random_forest", "rf"):
        return RandomForestClassifier(random_state=random_state, n_estimators=300)

    raise ValueError(f"Unknown model_name: {model_name}")


def build_model_pipeline(preprocessor: Pipeline, estimator) -> Pipeline:
    """Combine preprocessing + model into one Pipeline (prevents leakage)."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def tune_with_gridsearch(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    options: Optional[TrainOptions] = None,
):
    """Hyperparameter tuning using GridSearchCV with StratifiedKFold."""
    options = options or TrainOptions()

    cv = StratifiedKFold(
        n_splits=options.cv_splits,
        shuffle=True,
        random_state=options.random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=options.scoring,
        cv=cv,
        n_jobs=options.n_jobs,
        verbose=options.verbose,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def predict_with_optional_proba(model: Pipeline, X_test):
    """Predict labels and, if available, probability for positive class."""
    y_pred = model.predict(X_test)
    y_proba = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba is not None and proba.shape[1] >= 2:
                y_proba = proba[:, 1]
        except Exception:
            y_proba = None

    return y_pred, y_proba


def evaluate_on_test(model: Pipeline, X_test, y_test) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on test data (JSON-safe metrics)."""
    y_pred, y_proba = predict_with_optional_proba(model, X_test)
    return compute_classification_metrics(y_test, y_pred, y_proba=y_proba)
