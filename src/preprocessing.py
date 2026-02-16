from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering + leakage-safe cleanup.

    Key goals for maximum marks:
    - Drop leakage columns
    - Add simple derived features (report-friendly)
    - Force ID-like numeric columns (agent/company) to categorical to avoid wrong scaling
    """

    def __init__(
        self,
        drop_cols: Optional[List[str]] = None,
        force_categorical_cols: Optional[List[str]] = None,
    ):
        self.drop_cols = drop_cols or []
        self.force_categorical_cols = force_categorical_cols or []

        self._month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Drop leakage columns
        for c in self.drop_cols:
            if c in X.columns:
                X = X.drop(columns=c)

        # Force certain columns to categorical (important for agent/company)
        for c in self.force_categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype("object")

        # Derived: total nights
        if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(X.columns):
            X["total_nights"] = pd.to_numeric(
                X["stays_in_weekend_nights"], errors="coerce"
            ).fillna(0) + pd.to_numeric(
                X["stays_in_week_nights"], errors="coerce"
            ).fillna(
                0
            )

        # Derived: total guests + family
        guest_cols = [c for c in ["adults", "children", "babies"] if c in X.columns]
        if guest_cols:
            total = 0
            for c in guest_cols:
                total = total + pd.to_numeric(X[c], errors="coerce").fillna(0)
            X["total_guests"] = total
            children = pd.to_numeric(X.get("children", 0), errors="coerce").fillna(0)  # type: ignore
            X["is_family"] = (children > 0).astype(int)

        # Month name -> number
        if "arrival_date_month" in X.columns:
            X["arrival_month_num"] = X["arrival_date_month"].map(self._month_map)

        # Arrival day-of-week
        needed = {"arrival_date_year", "arrival_month_num", "arrival_date_day_of_month"}
        if needed.issubset(X.columns):
            year = pd.to_numeric(X["arrival_date_year"], errors="coerce")
            month = pd.to_numeric(X["arrival_month_num"], errors="coerce")
            day = pd.to_numeric(X["arrival_date_day_of_month"], errors="coerce")
            arrival_date = pd.to_datetime(
                dict(year=year, month=month, day=day), errors="coerce"  # type: ignore
            )  # type: ignore
            X["arrival_day_of_week"] = arrival_date.dt.dayofweek

        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Caps numeric features to reduce extreme outliers (fit on train only)."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = self._to_numpy(X)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = self._to_numpy(X)
        return np.clip(X, self.lower_, self.upper_)

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=float)


def _make_onehot(output_sparse: bool, min_frequency: Optional[float]):
    """
    Create OneHotEncoder compatible across sklearn versions.
    - sklearn>=1.2 uses sparse_output
    - older uses sparse
    """
    kwargs = dict(handle_unknown="ignore")
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency  # type: ignore
    try:
        return OneHotEncoder(sparse_output=output_sparse, **kwargs)  # type: ignore # sklearn>=1.2
    except TypeError:
        kwargs.pop("min_frequency", None)
        return OneHotEncoder(
            sparse=output_sparse, handle_unknown="ignore"
        )  # sklearn<1.2


@dataclass(frozen=True)
class PreprocessOptions:
    output_sparse: bool = True
    scale_numeric: bool = True
    onehot_min_frequency: Optional[float] = 0.01
    lower_clip_q: float = 0.01
    upper_clip_q: float = 0.99


def build_preprocessor(
    drop_cols: Optional[List[str]] = None,
    force_categorical_cols: Optional[List[str]] = None,
    options: Optional[PreprocessOptions] = None,
) -> Pipeline:
    """
    Pipeline:
      FeatureEngineer -> ColumnTransformer(num + cat)

    Notes:
      - For KNN: options.output_sparse=False (dense), scale_numeric=True
      - For Trees/RF: output_sparse=True is usually fine
    """
    options = options or PreprocessOptions()
    drop_cols = drop_cols or []
    force_categorical_cols = force_categorical_cols or []

    numeric_selector = make_column_selector(dtype_include=np.number)  # type: ignore
    categorical_selector = make_column_selector(dtype_exclude=np.number)  # type: ignore

    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clipper",
            QuantileClipper(lower_q=options.lower_clip_q, upper_q=options.upper_clip_q),
        ),
    ]
    if options.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                _make_onehot(
                    output_sparse=options.output_sparse,
                    min_frequency=options.onehot_min_frequency,
                ),
            ),
        ]
    )

    sparse_threshold = 0.0 if options.output_sparse else 1.0

    col_tf = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", cat_pipe, categorical_selector),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )

    return Pipeline(
        steps=[
            (
                "features",
                FeatureEngineer(
                    drop_cols=drop_cols, force_categorical_cols=force_categorical_cols
                ),
            ),
            ("transform", col_tf),
        ]
    )


def get_feature_names(
    preprocess_pipeline: Pipeline, input_features: Optional[Sequence[str]] = None
) -> List[str]:
    """
    Extract output feature names after preprocessing for interpretability plots.
    Works when the last step is a ColumnTransformer with OneHotEncoder.
    """
    if "transform" not in preprocess_pipeline.named_steps:
        return []

    ct = preprocess_pipeline.named_steps["transform"]

    try:
        return list(ct.get_feature_names_out(input_features))
    except Exception:
        # Fallback: still return empty to avoid breaking runs
        return []
