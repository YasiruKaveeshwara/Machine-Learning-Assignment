from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List


# =========================
# Project-wide configuration
# =========================

PROJECT_NAME: str = "ML-assignment"

# Reproducibility
RANDOM_STATE: int = 42

# Splits / CV
TEST_SIZE: float = 0.20
CV_SPLITS: int = 5

# Dataset / target
TARGET_COL: str = "is_canceled"

# Default dataset path (your notebooks can override)
DEFAULT_DATA_PATH: str = "data/raw/hotel_bookings.csv"

# Leakage-prone columns (contain post-booking outcomes)
LEAKAGE_COLS: List[str] = [
    "reservation_status",
    "reservation_status_date",
]

# Columns that look numeric but are actually IDs/categories (treat as categorical)
# This is important for good performance and correct preprocessing.
FORCE_CATEGORICAL_COLS: List[str] = ["agent", "company"]

# Feature engineering output columns (for reference/report)
DERIVED_FEATURES: List[str] = [
    "total_nights",
    "total_guests",
    "is_family",
    "arrival_month_num",
    "arrival_day_of_week",
]

# One-hot encoding controls
# min_frequency groups rare categories (reduces dimensionality); set None to disable.
ONEHOT_MIN_FREQUENCY: Optional[float] = 0.01  # e.g., 1%

# Plot saving quality
FIG_DPI: int = 160

# Where to store outputs (runs)
RUNS_DIR: str = "artifacts/runs"


# =========================
# Paths
# =========================
@dataclass(frozen=True)
class Paths:
    """Project paths relative to repository root."""

    root: Path
    data_raw: Path
    data_processed: Path
    artifacts: Path
    figures: Path
    runs: Path

    @staticmethod
    def from_repo_root(repo_root: Optional[Path] = None) -> "Paths":
        if repo_root is None:
            # src/config.py -> repo_root is parent of "src"
            repo_root = Path(__file__).resolve().parents[1]
        return Paths(
            root=repo_root,
            data_raw=repo_root / "data" / "raw",
            data_processed=repo_root / "data" / "processed",
            artifacts=repo_root / "artifacts",
            figures=repo_root / "figures",
            runs=repo_root / RUNS_DIR,
        )


# =========================
# Model grids (tuning)
# =========================
# In train_eval we build pipeline(preprocess + model),
# so params must be prefixed with "model__"

PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "logreg": {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["liblinear"],
        "model__max_iter": [2000],
    },
    "knn": {
        "model__n_neighbors": [5, 11, 21, 31],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["minkowski"],
        "model__p": [1, 2],  # 1=Manhattan, 2=Euclidean
    },
    "decision_tree": {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 10, 30],
        "model__min_samples_leaf": [1, 5, 10],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced"],
    },
}
