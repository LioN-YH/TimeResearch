"""QuitoBench common-window oracle target audit.

该工具只读取已有 expert cache，并从 registry + 原始数据重新抽取 target。
它不训练模型，不生成新专家预测，目标是判断 SNaive/DLinear/PatchTST 在同一批
physical_window_id 上是否有足够互补性，是否值得进入 visual gate smoke。
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

from tools.quitobench_common import ensure_unique_key, filter_common_expert_windows, prediction_columns, require_columns, write_json_manifest
from tools.quitobench_expert_cache_comparison import build_true_uniform_ensemble_metrics
from tools.quitobench_lightweight_expert_cache import DEFAULT_DATA_DIR, extract_histories_and_targets, load_registry


DEFAULT_SAMPLE_SET_ID = "qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506"
DEFAULT_REGISTRY_DIR = Path("outputs/vision_ts_routing/window_registry") / DEFAULT_SAMPLE_SET_ID
DEFAULT_EXPERT_ROOT = Path("outputs/vision_ts_routing/expert_predictions") / DEFAULT_SAMPLE_SET_ID
DEFAULT_OUTPUT_ROOT = Path("outputs/vision_ts_routing/oracle_audit") / DEFAULT_SAMPLE_SET_ID
DEFAULT_REQUIRED_EXPERTS = ("seasonal_naive", "dlinear_quito", "patchtst_quito")
DEFAULT_CACHE_DIRS = (
    DEFAULT_EXPERT_ROOT / "seasonal_naive_period6__official_align_h96_p48_allch_stride288_50k",
    DEFAULT_EXPERT_ROOT / "dlinear__official_align_h96_p48_allch_stride288_50k_e5_std",
    DEFAULT_EXPERT_ROOT / "patchtst__official_align_h96_p48_allch_stride288_50k_e5_std",
)


def _filter_common_keys(frame: pd.DataFrame, required_experts: Sequence[str]) -> pd.DataFrame:
    return filter_common_expert_windows(frame, required_experts)


def load_cache_tables(cache_dirs: Sequence[Path], required_experts: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取多个 cache 的 predictions/errors，并校验 expert-window key 唯一。"""

    prediction_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    required = set(required_experts)
    for cache_dir in cache_dirs:
        predictions_path = cache_dir / "predictions.parquet"
        errors_path = cache_dir / "errors.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError(f"缺少 predictions.parquet：{predictions_path}")
        if not errors_path.exists():
            raise FileNotFoundError(f"缺少 errors.parquet：{errors_path}")
        predictions = pd.read_parquet(predictions_path)
        errors = pd.read_parquet(errors_path)
        require_columns(predictions, {"physical_window_id", "expert_id"}, label=str(predictions_path))
        require_columns(errors, {"physical_window_id", "expert_id", "mse", "mae"}, label=str(errors_path))
        predictions = predictions[predictions["expert_id"].astype(str).isin(required)].copy()
        errors = errors[errors["expert_id"].astype(str).isin(required)].copy()
        prediction_frames.append(predictions)
        error_frames.append(errors)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    errors = pd.concat(error_frames, ignore_index=True)
    ensure_unique_key(predictions, ["physical_window_id", "expert_id"], label="predictions 合并后")
    ensure_unique_key(errors, ["physical_window_id", "expert_id"], label="errors 合并后")
    return predictions, errors


def _soft_oracle_entropy(errors: pd.DataFrame, temperature: float = 1.0, eps: float = 1e-8) -> float:
    if errors.empty:
        return float("nan")
    entropies: list[float] = []
    for _, group in errors.groupby("physical_window_id", sort=False):
        scores = -group["mse"].to_numpy(dtype=float) / max(float(temperature), eps)
        scores = scores - float(np.max(scores))
        weights = np.exp(scores)
        weights = weights / max(float(weights.sum()), eps)
        entropies.append(float(-(weights * np.log(weights + eps)).sum()))
    return float(np.mean(entropies))


def _summary_for_group(
    errors: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
    required_experts: Sequence[str],
    group_col: str | None = None,
    group_value: str | None = None,
) -> dict[str, object]:
    prefix = {group_col: group_value} if group_col else {}
    if errors.empty:
        return {
            **prefix,
            "num_common_windows": 0,
            "num_experts": 0,
            "best_fixed_expert": "",
            "best_fixed_mse": np.nan,
            "best_fixed_mae": np.nan,
            "true_uniform_mse": np.nan,
            "true_uniform_mae": np.nan,
            "oracle_top1_mse": np.nan,
            "oracle_top1_mae": np.nan,
            "oracle_gap_vs_best_fixed": np.nan,
            "oracle_gap_vs_true_uniform": np.nan,
            "soft_oracle_entropy": np.nan,
        }

    common_ids = set(errors["physical_window_id"].astype(str).unique())
    pred_group = predictions[predictions["physical_window_id"].astype(str).isin(common_ids)].copy()
    target_group = {key: targets[key] for key in common_ids if key in targets}
    uniform = build_true_uniform_ensemble_metrics(pred_group, target_group, required_experts=required_experts)
    fixed = errors.groupby("expert_id").agg(mse=("mse", "mean"), mae=("mae", "mean")).sort_values(["mse", "mae"])
    best_fixed_expert = str(fixed.index[0])
    best_fixed_mse = float(fixed.iloc[0]["mse"])
    oracle_by_window = errors.loc[errors.groupby("physical_window_id")["mse"].idxmin()]
    oracle_mse = float(oracle_by_window["mse"].mean())
    return {
        **prefix,
        "num_common_windows": int(errors["physical_window_id"].nunique()),
        "num_experts": int(errors["expert_id"].nunique()),
        "best_fixed_expert": best_fixed_expert,
        "best_fixed_mse": best_fixed_mse,
        "best_fixed_mae": float(fixed.iloc[0]["mae"]),
        "true_uniform_mse": float(uniform.loc[0, "true_uniform_mse"]),
        "true_uniform_mae": float(uniform.loc[0, "true_uniform_mae"]),
        "oracle_top1_mse": oracle_mse,
        "oracle_top1_mae": float(oracle_by_window["mae"].mean()),
        "oracle_gap_vs_best_fixed": best_fixed_mse - oracle_mse,
        "oracle_gap_vs_true_uniform": float(uniform.loc[0, "true_uniform_mse"]) - oracle_mse,
        "soft_oracle_entropy": _soft_oracle_entropy(errors),
    }


def _group_summary(
    errors: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
    required_experts: Sequence[str],
    group_col: str,
) -> pd.DataFrame:
    if group_col not in errors.columns:
        return pd.DataFrame()
    rows = [
        _summary_for_group(
            group,
            predictions,
            targets,
            required_experts=required_experts,
            group_col=group_col,
            group_value=str(value),
        )
        for value, group in errors.groupby(group_col, sort=True)
    ]
    return pd.DataFrame(rows)


def _expert_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["expert_id", "mse", "mae", "oracle_top1_rate", "num_windows"])
    ranked = errors.copy()
    ranked["rank_in_window"] = ranked.groupby("physical_window_id")["mse"].rank(method="first", ascending=True).astype(int)
    ranked["is_oracle_top1"] = ranked["rank_in_window"] == 1
    return (
        ranked.groupby("expert_id", as_index=False)
        .agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            oracle_top1_rate=("is_oracle_top1", "mean"),
            num_windows=("physical_window_id", "nunique"),
        )
        .sort_values(["mse", "expert_id"])
        .reset_index(drop=True)
    )


def build_oracle_target_audit(
    errors: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
    required_experts: Sequence[str] = DEFAULT_REQUIRED_EXPERTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算 common-window best fixed、true uniform、oracle top1 和分组指标。"""

    yhat_cols = prediction_columns(predictions)
    if not yhat_cols:
        raise ValueError("predictions 缺少 yhat_* 预测列")
    common_errors = _filter_common_keys(errors, required_experts)
    common_predictions = _filter_common_keys(predictions, required_experts)
    shared_ids = set(common_errors["physical_window_id"].astype(str)) & set(common_predictions["physical_window_id"].astype(str)) & set(targets)
    common_errors = common_errors[common_errors["physical_window_id"].astype(str).isin(shared_ids)].copy()
    common_predictions = common_predictions[common_predictions["physical_window_id"].astype(str).isin(shared_ids)].copy()

    summary = pd.DataFrame([_summary_for_group(common_errors, common_predictions, targets, required_experts=required_experts)])
    by_split = _group_summary(common_errors, common_predictions, targets, required_experts, "split")
    by_subset = _group_summary(common_errors, common_predictions, targets, required_experts, "subset")
    by_cell = _group_summary(common_errors, common_predictions, targets, required_experts, "official_tsf_cell")
    expert_metrics = _expert_metrics(common_errors)
    return summary, by_split, by_subset, by_cell, expert_metrics


def write_oracle_target_audit_outputs(
    summary: pd.DataFrame,
    by_split: pd.DataFrame,
    by_subset: pd.DataFrame,
    by_cell: pd.DataFrame,
    expert_metrics: pd.DataFrame,
    manifest: Mapping[str, object],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "oracle_summary.csv", index=False)
    by_split.to_csv(output_dir / "oracle_by_split.csv", index=False)
    by_subset.to_csv(output_dir / "oracle_by_subset.csv", index=False)
    by_cell.to_csv(output_dir / "oracle_by_cell.csv", index=False)
    expert_metrics.to_csv(output_dir / "expert_metrics.csv", index=False)
    write_json_manifest(output_dir / "manifest.json", manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--required-experts", default=",".join(DEFAULT_REQUIRED_EXPERTS))
    parser.add_argument("--audit-id", default="common_23456_official_align")
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
    _, targets = extract_histories_and_targets(registry, data_dir=args.data_dir, progress_every=args.progress_every)
    summary, by_split, by_subset, by_cell, expert_metrics = build_oracle_target_audit(
        errors=errors,
        predictions=predictions,
        targets=targets,
        required_experts=required_experts,
    )
    manifest = {
        "stage": "canonical_oracle_target_audit",
        "audit_id": args.audit_id,
        "required_experts": list(required_experts),
        "cache_dirs": [str(path) for path in cache_dirs],
        "registry_dir": str(args.registry_dir),
        "data_dir": str(args.data_dir),
        "sample_set_id": registry_manifest.get("sample_set_id", DEFAULT_SAMPLE_SET_ID),
        "num_common_windows": int(summary.loc[0, "num_common_windows"]) if not summary.empty else 0,
        "metric_scale": "raw_scale",
        "true_uniform_definition": "average predictions first, then compute MSE/MAE against raw target",
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
