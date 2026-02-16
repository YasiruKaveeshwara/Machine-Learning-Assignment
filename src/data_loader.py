from __future__ import annotations

from pathlib import Path
from typing import Union, Dict, Any, Optional

import pandas as pd


def load_hotel_bookings(
    csv_path: Union[str, Path],
    drop_duplicates: bool = True,
    verbose: bool = True,
    na_values: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load the Hotel Booking Demand dataset from CSV.

    This function intentionally does ONLY safe loading steps:
    - read CSV
    - strip column name whitespace
    - optionally drop exact duplicate rows
    - ensure target column dtype is int (0/1)

    Anything that can cause leakage (imputation/encoding/scaling) must be done
    in the preprocessing pipeline (preprocessing.py) using train-only fitting.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    if na_values is None:
        # common missing tokens (CSV usually already has blanks as NaN)
        na_values = ["NA", "N/A", ""]

    df = pd.read_csv(csv_path, na_values=na_values)

    # Clean column names (keep original casing to match dataset docs)
    df.columns = [c.strip() for c in df.columns]

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if verbose:
            print(f"[data_loader] Dropped duplicates: {before - after:,} rows")

    # Ensure target is integer 0/1 if present
    if "is_canceled" in df.columns:
        df["is_canceled"] = df["is_canceled"].astype(int)

    if verbose:
        print(f"[data_loader] Loaded shape: {df.shape}")
        print(f"[data_loader] Columns: {len(df.columns)}")

    return df


def basic_train_ready_checks(df: pd.DataFrame, target_col: str = "is_canceled") -> None:
    """Quick sanity checks (helps avoid silent issues)."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values.")
    unique = df[target_col].dropna().unique()
    if set(unique) - {0, 1}:
        raise ValueError(f"Target column must be binary 0/1. Found: {sorted(unique)}")


def summarize_dataframe(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a JSON-friendly summary you can save in every run:
    - shape, columns, dtypes
    - missing counts
    - target distribution if target_col provided
    """
    summary: Dict[str, Any] = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_count": {c: int(df[c].isna().sum()) for c in df.columns},
    }
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False).to_dict()
        summary["target_value_counts"] = {str(k): int(v) for k, v in vc.items()}
        summary["target_positive_rate"] = float(df[target_col].mean())
    return summary
