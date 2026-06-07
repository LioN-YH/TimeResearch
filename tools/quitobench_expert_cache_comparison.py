"""Stage 1.4b：同一窗口集合上的专家缓存互补性汇总。

读取多个 expert cache 的 `errors.parquet`，按共同 `physical_window_id`
计算 best fixed expert、uniform proxy、oracle ensemble 和 oracle gap。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_SET_ID = "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e"
DEFAULT_EXPERT_ROOT = Path("outputs/vision_ts_routing/expert_predictions") / DEFAULT_SAMPLE_SET_ID
DEFAULT_OUTPUT_ROOT = Path("outputs/vision_ts_routing/expert_comparisons") / DEFAULT_SAMPLE_SET_ID
DEFAULT_REQUIRED_EXPERTS = ("seasonal_naive", "dlinear_quito", "patchtst_quito")


def load_error_tables(cache_dirs: Sequence[Path], required_experts: Sequence[str]) -> pd.DataFrame:
    """读取多个 cache 目录中的 errors，并只保留指定专家。"""

    frames: list[pd.DataFrame] = []
    required = set(required_experts)
    for cache_dir in cache_dirs:
        path = cache_dir / "errors.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少 errors.parquet：{path}")
        frame = pd.read_parquet(path)
        missing = {"physical_window_id", "expert_id", "mse", "mae"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少列：{sorted(missing)}")
        frame = frame[frame["expert_id"].astype(str).isin(required)].copy()
        frames.append(frame)
    errors = pd.concat(frames, ignore_index=True)
    if errors[["physical_window_id", "expert_id"]].duplicated().any():
        duplicated = errors[errors[["physical_window_id", "expert_id"]].duplicated()][["physical_window_id", "expert_id"]].head()
        raise ValueError(f"合并后存在重复 expert-window 键：{duplicated.to_dict(orient='records')}")
    return errors


def _filter_common_windows(errors: pd.DataFrame, required_experts: Sequence[str]) -> pd.DataFrame:
    required = set(required_experts)
    filtered = errors[errors["expert_id"].astype(str).isin(required)].copy()
    expert_counts = filtered.groupby("physical_window_id")["expert_id"].nunique()
    common_ids = expert_counts[expert_counts == len(required)].index
    return filtered[filtered["physical_window_id"].isin(common_ids)].copy()


def _summary_for_group(errors: pd.DataFrame, group_value: str | None = None, group_col: str | None = None) -> dict[str, object]:
    if errors.empty:
        return {
            **({group_col: group_value} if group_col else {}),
            "num_common_windows": 0,
            "num_experts": 0,
            "oracle_mse": np.nan,
            "best_fixed_expert": "",
            "best_fixed_mse": np.nan,
            "uniform_mse_proxy": np.nan,
            "oracle_gap_vs_best_fixed": np.nan,
        }
    oracle_mse = float(errors.groupby("physical_window_id")["mse"].min().mean())
    fixed = errors.groupby("expert_id")["mse"].mean().sort_values()
    best_fixed_expert = str(fixed.index[0])
    best_fixed_mse = float(fixed.iloc[0])
    uniform_proxy = float(errors.groupby("physical_window_id")["mse"].mean().mean())
    return {
        **({group_col: group_value} if group_col else {}),
        "num_common_windows": int(errors["physical_window_id"].nunique()),
        "num_experts": int(errors["expert_id"].nunique()),
        "oracle_mse": oracle_mse,
        "best_fixed_expert": best_fixed_expert,
        "best_fixed_mse": best_fixed_mse,
        "uniform_mse_proxy": uniform_proxy,
        "oracle_gap_vs_best_fixed": best_fixed_mse - oracle_mse,
    }


def _group_summary(errors: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in errors.groupby(group_col, sort=True):
        rows.append(_summary_for_group(group, group_value=str(value), group_col=group_col))
    return pd.DataFrame(rows)


def build_expert_comparison(
    errors: pd.DataFrame,
    required_experts: Sequence[str] = DEFAULT_REQUIRED_EXPERTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """基于共同窗口计算 overall、split 和 cell 三个层级的 oracle gap。"""

    common = _filter_common_windows(errors, required_experts)
    summary = pd.DataFrame([_summary_for_group(common)])
    by_split = _group_summary(common, "split") if "split" in common.columns else pd.DataFrame()
    by_cell = _group_summary(common, "official_tsf_cell") if "official_tsf_cell" in common.columns else pd.DataFrame()
    return summary, by_split, by_cell


def build_expert_metrics(errors: pd.DataFrame, required_experts: Sequence[str] = DEFAULT_REQUIRED_EXPERTS) -> pd.DataFrame:
    common = _filter_common_windows(errors, required_experts)
    if common.empty:
        return pd.DataFrame(columns=["expert_id", "mse", "mae", "oracle_top1_rate", "num_windows"])
    ranked = common.copy()
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


def write_comparison_outputs(
    summary: pd.DataFrame,
    by_split: pd.DataFrame,
    by_cell: pd.DataFrame,
    expert_metrics: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    by_split.to_csv(output_dir / "comparison_by_split.csv", index=False)
    by_cell.to_csv(output_dir / "comparison_by_cell.csv", index=False)
    expert_metrics.to_csv(output_dir / "expert_metrics.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        required=True,
        help="包含 errors.parquet 的 expert_set 目录，可重复传入。",
    )
    parser.add_argument("--required-experts", default=",".join(DEFAULT_REQUIRED_EXPERTS))
    parser.add_argument("--comparison-id", default="seasonal_naive_dlinear_patchtst__stratified_smoke_5k")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_experts = tuple(expert.strip() for expert in args.required_experts.split(",") if expert.strip())
    errors = load_error_tables(args.cache_dir, required_experts)
    summary, by_split, by_cell = build_expert_comparison(errors, required_experts=required_experts)
    expert_metrics = build_expert_metrics(errors, required_experts=required_experts)
    manifest = {
        "stage": "stage1_4b_expert_cache_comparison",
        "comparison_id": args.comparison_id,
        "required_experts": list(required_experts),
        "cache_dirs": [str(path) for path in args.cache_dir],
        "num_common_windows": int(summary.loc[0, "num_common_windows"]) if not summary.empty else 0,
        "implements_router": False,
        "runs_visual_encoder": False,
        "output_files": {
            "summary": "comparison_summary.csv",
            "by_split": "comparison_by_split.csv",
            "by_cell": "comparison_by_cell.csv",
            "expert_metrics": "expert_metrics.csv",
            "manifest": "manifest.json",
        },
    }
    out_dir = write_comparison_outputs(
        summary=summary,
        by_split=by_split,
        by_cell=by_cell,
        expert_metrics=expert_metrics,
        manifest=manifest,
        output_dir=args.output_root / args.comparison_id,
    )
    print(f"[done] output_dir={out_dir}")
    print(f"[done] common_windows={manifest['num_common_windows']}")
    print(f"[done] experts={','.join(required_experts)}")


if __name__ == "__main__":
    main()
