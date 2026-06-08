"""QuitoBench expert cache 审计工具。

只读取已有 `predictions.parquet`、`errors.parquet` 和 `manifest.json`，
检查 key 唯一性、horizon、manifest 字段和多专家 cache 的共同窗口覆盖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_common import load_json_manifest, prediction_columns, require_columns, write_json_manifest


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def audit_single_cache(cache_dir: Path) -> dict[str, object]:
    """审计单个 expert cache 目录。"""

    predictions_path = cache_dir / "predictions.parquet"
    errors_path = cache_dir / "errors.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"缺少 predictions.parquet：{predictions_path}")
    if not errors_path.exists():
        raise FileNotFoundError(f"缺少 errors.parquet：{errors_path}")

    predictions = pd.read_parquet(predictions_path)
    errors = pd.read_parquet(errors_path)
    manifest = load_json_manifest(cache_dir / "manifest.json")
    require_columns(predictions, {"physical_window_id", "expert_id"}, label=str(predictions_path))
    require_columns(errors, {"physical_window_id", "expert_id", "mse", "mae"}, label=str(errors_path))

    yhat_cols = prediction_columns(predictions)
    manifest_pred_len = dict(manifest.get("config") or {}).get("pred_len")
    expert_ids = sorted(set(predictions["expert_id"].astype(str)) | set(errors["expert_id"].astype(str)))
    prediction_ids = set(predictions["physical_window_id"].astype(str))
    error_ids = set(errors["physical_window_id"].astype(str))
    standardization = dict(manifest.get("standardization") or {})
    return {
        "cache_dir": str(cache_dir),
        "expert_set_id": manifest.get("expert_set_id", cache_dir.name),
        "expert_ids": expert_ids,
        "sample_set_id": _as_list(manifest.get("sample_set_id")),
        "base_registry_id": _as_list(manifest.get("base_registry_id")),
        "prediction_rows": int(len(predictions)),
        "error_rows": int(len(errors)),
        "unique_prediction_key": bool(not predictions[["physical_window_id", "expert_id"]].duplicated().any()),
        "unique_error_key": bool(not errors[["physical_window_id", "expert_id"]].duplicated().any()),
        "prediction_windows": int(len(prediction_ids)),
        "error_windows": int(len(error_ids)),
        "prediction_error_window_overlap": int(len(prediction_ids & error_ids)),
        "num_yhat_cols": int(len(yhat_cols)),
        "manifest_pred_len": manifest_pred_len,
        "horizon_matches_manifest": bool(manifest_pred_len is None or int(manifest_pred_len) == len(yhat_cols)),
        "future_read_policy": manifest.get("future_read_policy"),
        "target_usage": manifest.get("target_usage"),
        "runs_neural_experts": manifest.get("runs_neural_experts"),
        "train_set_standardize": dict(manifest.get("config") or {}).get("train_set_standardize"),
        "standardization_enabled": standardization.get("enabled"),
        "standardization_scope": standardization.get("scope"),
    }


def audit_cache_collection(cache_dirs: Sequence[Path]) -> tuple[dict[str, object], pd.DataFrame]:
    """审计多个 expert cache，并计算共同窗口交集。"""

    if not cache_dirs:
        raise ValueError("cache_dirs 不能为空")
    rows = [audit_single_cache(cache_dir) for cache_dir in cache_dirs]
    prediction_sets: list[set[str]] = []
    error_sets: list[set[str]] = []
    expert_ids: set[str] = set()
    for cache_dir, row in zip(cache_dirs, rows, strict=True):
        predictions = pd.read_parquet(cache_dir / "predictions.parquet")
        errors = pd.read_parquet(cache_dir / "errors.parquet")
        prediction_sets.append(set(predictions["physical_window_id"].astype(str)))
        error_sets.append(set(errors["physical_window_id"].astype(str)))
        expert_ids.update(str(expert_id) for expert_id in row["expert_ids"])

    common_prediction = set.intersection(*prediction_sets) if prediction_sets else set()
    common_error = set.intersection(*error_sets) if error_sets else set()
    summary = {
        "num_caches": int(len(cache_dirs)),
        "expert_ids": sorted(expert_ids),
        "common_prediction_windows": int(len(common_prediction)),
        "common_error_windows": int(len(common_error)),
        "all_horizons_match_manifest": bool(all(row["horizon_matches_manifest"] for row in rows)),
        "all_prediction_keys_unique": bool(all(row["unique_prediction_key"] for row in rows)),
        "all_error_keys_unique": bool(all(row["unique_error_key"] for row in rows)),
    }
    per_cache = pd.DataFrame(rows)
    return summary, per_cache


def write_cache_audit_outputs(
    summary: dict[str, object],
    per_cache: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """写出 expert cache audit 结果。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_manifest(output_dir / "cache_audit_summary.json", summary)
    per_cache.to_csv(output_dir / "cache_audit_per_cache.csv", index=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, per_cache = audit_cache_collection(args.cache_dir)
    out_dir = write_cache_audit_outputs(summary, per_cache, args.output_dir)
    print(f"[done] output_dir={out_dir}")
    print(f"[done] expert_ids={summary['expert_ids']}")
    print(f"[done] common_prediction_windows={summary['common_prediction_windows']}")
    print(f"[done] common_error_windows={summary['common_error_windows']}")


if __name__ == "__main__":
    main()
