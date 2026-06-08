"""Small shared helpers for QuitoBench tooling.

Only place low-risk, side-effect-light helpers here. Keep data loading,
standardization, and model training logic in their owning scripts until those
paths are fully stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def prediction_columns(frame: pd.DataFrame, prefix: str = "yhat_") -> list[str]:
    """Return wide prediction columns sorted by horizon index."""

    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("_", 1)[1]))


def load_json_manifest(path: Path) -> dict[str, object]:
    """Load a JSON manifest, returning an empty dict if the file is absent."""

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    """Write a JSON manifest with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def require_columns(frame: pd.DataFrame, required_columns: set[str], label: str = "frame") -> None:
    """Raise if a DataFrame is missing required columns."""

    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} 缺少列：{sorted(missing)}")


def ensure_unique_key(frame: pd.DataFrame, key_columns: Sequence[str], label: str = "frame") -> None:
    """Raise if a DataFrame has duplicate rows for the given key columns."""

    require_columns(frame, set(key_columns), label=label)
    duplicated = frame[list(key_columns)].duplicated()
    if duplicated.any():
        preview = frame.loc[duplicated, list(key_columns)].head()
        raise ValueError(f"{label} 存在重复键：{preview.to_dict(orient='records')}")


def filter_common_expert_windows(frame: pd.DataFrame, required_experts: Sequence[str]) -> pd.DataFrame:
    """Keep rows whose physical_window_id has all required experts."""

    require_columns(frame, {"physical_window_id", "expert_id"})
    required = set(required_experts)
    filtered = frame[frame["expert_id"].astype(str).isin(required)].copy()
    expert_counts = filtered.groupby("physical_window_id")["expert_id"].nunique()
    common_ids = expert_counts[expert_counts == len(required)].index
    return filtered[filtered["physical_window_id"].isin(common_ids)].copy()
