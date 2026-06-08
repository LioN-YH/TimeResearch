"""QuitoBench normalized-scale oracle target audit.

This is a read-only audit over existing raw-scale expert caches. It rebuilds the
Quito train-segment scaler for each registry window, transforms raw predictions
and targets to normalized scale, then reuses the common oracle audit summaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_common import prediction_columns, require_columns
from tools.quitobench_framework_expert_cache import QuitoWindowScaler, extract_quito_standardized_series_maps
from tools.quitobench_lightweight_expert_cache import DEFAULT_DATA_DIR, load_registry
from tools.quitobench_oracle_target_audit import (
    DEFAULT_CACHE_DIRS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_DIR,
    DEFAULT_REQUIRED_EXPERTS,
    _filter_common_keys,
    build_oracle_target_audit,
    load_cache_tables,
    write_oracle_target_audit_outputs,
)


def build_normalized_prediction_error_tables(
    predictions: pd.DataFrame,
    *,
    normalized_targets: Mapping[str, Sequence[float]],
    scalers_by_id: Mapping[str, QuitoWindowScaler],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform raw prediction rows to normalized scale and recompute errors."""

    require_columns(predictions, {"physical_window_id", "expert_id"}, label="predictions")
    yhat_cols = prediction_columns(predictions)
    if not yhat_cols:
        raise ValueError("predictions 缺少 yhat_* 预测列")

    normalized_predictions = predictions.copy()
    error_rows: list[dict[str, object]] = []
    normalized_values: list[np.ndarray] = []
    metadata_cols = [
        col
        for col in ["physical_window_id", "expert_id", "split", "subset", "official_tsf_cell"]
        if col in predictions.columns
    ]

    for _, row in normalized_predictions.iterrows():
        physical_window_id = str(row["physical_window_id"])
        if physical_window_id not in scalers_by_id:
            raise KeyError(f"缺少 scaler：{physical_window_id}")
        if physical_window_id not in normalized_targets:
            raise KeyError(f"缺少 normalized target：{physical_window_id}")
        raw_pred = row[yhat_cols].to_numpy(dtype=np.float32)
        norm_pred = scalers_by_id[physical_window_id].transform(raw_pred).astype(float)
        norm_target = np.asarray(normalized_targets[physical_window_id], dtype=float)
        if len(norm_pred) != len(norm_target):
            raise ValueError(f"{physical_window_id} prediction/target horizon 不一致")
        normalized_values.append(norm_pred)
        diff = norm_pred - norm_target
        error_row = {col: row[col] for col in metadata_cols}
        error_row["mse"] = float(np.mean(diff**2))
        error_row["mae"] = float(np.mean(np.abs(diff)))
        error_rows.append(error_row)

    normalized_predictions.loc[:, yhat_cols] = np.vstack(normalized_values)
    return normalized_predictions, pd.DataFrame(error_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--required-experts", default=",".join(DEFAULT_REQUIRED_EXPERTS))
    parser.add_argument("--audit-id", default="matrix50k_v1_normalized")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dirs = tuple(args.cache_dir or DEFAULT_CACHE_DIRS)
    required_experts = tuple(expert.strip() for expert in args.required_experts.split(",") if expert.strip())
    predictions, errors = load_cache_tables(cache_dirs, required_experts=required_experts)
    common_ids = sorted(set(_filter_common_keys(errors, required_experts)["physical_window_id"].astype(str)))
    registry, registry_manifest = load_registry(args.registry_dir)
    registry = registry[registry["physical_window_id"].astype(str).isin(common_ids)].copy()
    _, normalized_targets, _, scalers_by_id, standardization = extract_quito_standardized_series_maps(
        registry,
        data_dir=args.data_dir,
        progress_every=args.progress_every,
    )
    normalized_predictions, normalized_errors = build_normalized_prediction_error_tables(
        predictions[predictions["physical_window_id"].astype(str).isin(common_ids)].copy(),
        normalized_targets=normalized_targets,
        scalers_by_id=scalers_by_id,
    )
    summary, by_split, by_subset, by_cell, expert_metrics = build_oracle_target_audit(
        errors=normalized_errors,
        predictions=normalized_predictions,
        targets=normalized_targets,
        required_experts=required_experts,
    )
    manifest = {
        "stage": "canonical_normalized_oracle_target_audit",
        "audit_id": args.audit_id,
        "required_experts": list(required_experts),
        "cache_dirs": [str(path) for path in cache_dirs],
        "registry_dir": str(args.registry_dir),
        "data_dir": str(args.data_dir),
        "sample_set_id": registry_manifest.get("sample_set_id"),
        "num_common_windows": int(summary.loc[0, "num_common_windows"]) if not summary.empty else 0,
        "metric_scale": "normalized_scale",
        "normalization": standardization,
        "prediction_transform": "raw cached predictions transformed with Quito train-segment subset/item/channel scaler",
        "target_transform": "targets transformed with the same Quito train-segment subset/item/channel scaler",
        "true_uniform_definition": "average normalized predictions first, then compute MSE/MAE against normalized target",
        "runs_training": False,
        "runs_visual_encoder": False,
        "output_files": {
            "summary": "oracle_summary.csv",
            "by_split": "oracle_by_split.csv",
            "by_subset": "oracle_by_subset.csv",
            "by_cell": "oracle_by_cell.csv",
            "expert_metrics": "expert_metrics.csv",
            "manifest": "manifest.json",
        },
    }
    out_dir = write_oracle_target_audit_outputs(
        summary,
        by_split,
        by_subset,
        by_cell,
        expert_metrics,
        manifest=manifest,
        output_dir=args.output_root / args.audit_id,
    )
    print(f"[done] output_dir={out_dir}")
    print(f"[done] common_windows={manifest['num_common_windows']}")
    print(f"[done] metric_scale={manifest['metric_scale']}")


if __name__ == "__main__":
    main()
