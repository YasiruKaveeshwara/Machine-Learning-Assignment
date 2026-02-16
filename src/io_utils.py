from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

PathLike = Union[str, Path]


# =========================
# Artifact directory helpers
# =========================


def ensure_artifact_dirs(base_dir: PathLike = "artifacts") -> Dict[str, Path]:
    """Create (if missing) and return standard artifact directories.

    This project uses *fixed* folders (no timestamps). Every run overwrites outputs:
      artifacts/
        data/
        preprocessing/
        models/
        metrics/
        plots/
        reports/
    """
    base = Path(base_dir)
    subdirs = {
        "base": base,
        "data": base / "data",
        "preprocessing": base / "preprocessing",
        "models": base / "models",
        "metrics": base / "metrics",
        "plots": base / "plots",
        "reports": base / "reports",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


# =========================
# Safe save/load utilities
# =========================


def _json_default(o: Any):
    if is_dataclass(o):
        return asdict(o)  # type: ignore
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically to reduce risk of partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_json(obj: Any, path: PathLike) -> None:
    """Save JSON (overwrites). Supports dataclasses and Paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def save_text(text: str, path: PathLike) -> None:
    """Save plain text (overwrites)."""
    _atomic_write_text(Path(path), text)


def save_dataframe(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Save DataFrame to CSV (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)


def save_model(model: Any, path: PathLike) -> None:
    """Save sklearn pipeline/model with joblib (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: PathLike) -> Any:
    """Load a saved model (joblib)."""
    return joblib.load(Path(path))


# =========================
# Repro metadata (optional)
# =========================


def get_git_commit_hash(repo_root: PathLike = ".") -> Optional[str]:
    """Return current git commit hash if available; else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def save_run_metadata(
    metadata: Dict[str, Any],
    base_dir: PathLike = "artifacts",
    repo_root: PathLike = ".",
    filename: str = "run_metadata.json",
) -> Path:
    """Save run metadata to artifacts/reports/<filename> (overwrites)."""
    dirs = ensure_artifact_dirs(base_dir)
    meta = dict(metadata)
    meta.setdefault("cwd", os.getcwd())
    meta.setdefault("git_commit", get_git_commit_hash(repo_root))
    out_path = dirs["reports"] / filename
    save_json(meta, out_path)
    return out_path
