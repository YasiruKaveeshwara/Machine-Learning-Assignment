from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd


PathLike = Union[str, Path]


def make_run_dir(
    base_dir: PathLike = "artifacts/runs", run_name: Optional[str] = None
) -> Path:
    """
    Create a unique run directory:
      artifacts/runs/YYYY-MM-DD_HH-MM-SS

    And standard subfolders:
      data/, preprocessing/, models/, metrics/, plots/, reports/
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for sub in ["data", "preprocessing", "models", "metrics", "plots", "reports"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir


def _json_default(o: Any):
    """Fallback serializer for non-JSON objects."""
    if is_dataclass(o):
        return asdict(o)  # type: ignore
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj: Any, path: PathLike) -> None:
    """Save any python object as JSON (dataclass-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_text(text: str, path: PathLike) -> None:
    """Save plain text to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Save DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_parquet(df: pd.DataFrame, path: PathLike) -> None:
    """Save DataFrame to parquet (smaller + faster), if pyarrow/fastparquet installed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_model(model: Any, path: PathLike) -> None:
    """Save sklearn pipeline/model safely using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: PathLike) -> Any:
    """Load a saved model (joblib)."""
    path = Path(path)
    return joblib.load(path)


def get_git_commit_hash(repo_root: PathLike = ".") -> Optional[str]:
    """
    Returns current git commit hash if available.
    Safe: returns None if not a git repo / git not installed.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def save_run_metadata(
    run_dir: PathLike,
    metadata: Dict[str, Any],
    repo_root: PathLike = ".",
) -> None:
    """
    Save run-level metadata (dataset path, seed, environment, git hash, etc.)
    into: <run_dir>/run_metadata.json
    """
    run_dir = Path(run_dir)
    meta = dict(metadata)

    meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    meta.setdefault("cwd", os.getcwd())
    meta.setdefault("git_commit", get_git_commit_hash(repo_root))

    save_json(meta, run_dir / "run_metadata.json")
