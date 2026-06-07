"""Stage 1.4e：专家预测尺度诊断工具。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tools.quitobench_lightweight_expert_cache import (
    DEFAULT_REGISTRY_DIR,
    extract_histories_and_targets,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315"


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p99": np.nan,
            "p999": np.nan,
        }
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "p999": float(np.quantile(finite, 0.999)),
    }


def _horizon_columns(predictions: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in predictions.columns if col.startswith("yhat_")],
        key=lambda value: int(value.split("_", 1)[1]),
    )


def summarize_prediction_scale(
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """汇总 prediction、target 和 absolute error 的尺度分布。"""

    horizon_cols = _horizon_columns(predictions)
    if not horizon_cols:
        raise ValueError("predictions 缺少 yhat_* 列")
    pred_values = predictions[horizon_cols].to_numpy(dtype=float)
    target_values = np.stack(
        [np.asarray(targets[str(row.physical_window_id)], dtype=float) for row in predictions.itertuples(index=False)]
    )
    if pred_values.shape != target_values.shape:
        raise ValueError(f"prediction shape {pred_values.shape} != target shape {target_values.shape}")
    abs_error = np.abs(pred_values - target_values)
    return {
        "rows": int(len(predictions)),
        "horizon_columns": int(len(horizon_cols)),
        "expert_ids": sorted(predictions["expert_id"].astype(str).unique().tolist()) if "expert_id" in predictions else [],
        "finite_prediction_rate": float(np.isfinite(pred_values).mean()),
        "prediction": _stats(pred_values),
        "target": _stats(target_values),
        "absolute_error": _stats(abs_error),
    }


def _load_targets_for_predictions(
    predictions: pd.DataFrame,
    registry_dir: Path,
    data_dir: Path,
) -> dict[str, np.ndarray]:
    registry, _ = load_registry(registry_dir)
    prediction_ids = set(predictions["physical_window_id"].astype(str))
    registry = registry[registry["physical_window_id"].astype(str).isin(prediction_ids)].copy()
    if len(registry) != len(prediction_ids):
        found_ids = set(registry["physical_window_id"].astype(str))
        missing = sorted(prediction_ids - found_ids)
        raise KeyError(f"registry 缺少 prediction ids：{missing[:5]}")
    _, targets = extract_histories_and_targets(registry, data_dir=data_dir)
    return {key: np.asarray(value, dtype=float) for key, value in targets.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.cache_dir / "predictions.parquet")
    targets = _load_targets_for_predictions(predictions, registry_dir=args.registry_dir, data_dir=args.data_dir)
    summary = summarize_prediction_scale(predictions, targets)
    output_json = args.output_json or (args.cache_dir / "prediction_diagnostics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] output_json={output_json}")


if __name__ == "__main__":
    main()
