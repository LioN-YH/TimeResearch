"""Stage 1.4a：QuitoBench sample-channel 轻量专家预测缓存。

本脚本只运行 history-only 的极轻量专家，生成专家预测、误差和 oracle
profiling 缓存。不训练视觉 encoder，不实现 router，不运行神经网络专家。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_window_registry import DEFAULT_DATA_DIR, load_subset_frames


DEFAULT_SAMPLE_SET_ID = "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e"
DEFAULT_REGISTRY_DIR = ROOT / "outputs/vision_ts_routing/window_registry" / DEFAULT_SAMPLE_SET_ID
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/expert_predictions" / DEFAULT_SAMPLE_SET_ID

EXPERT_IDS = ("last_value", "seasonal_naive", "recent_mean", "linear_trend")
EXPERT_FAMILY = {
    "last_value": "statistical_baseline",
    "seasonal_naive": "statistical_baseline",
    "recent_mean": "statistical_baseline",
    "linear_trend": "statistical_baseline",
}

REQUIRED_REGISTRY_COLUMNS = {
    "physical_window_id",
    "window_id",
    "base_registry_id",
    "sample_set_id",
    "subset",
    "split",
    "item_id",
    "channel",
    "period",
    "official_tsf_cell",
    "history_start_idx",
    "history_end_idx",
    "target_start_idx",
    "target_end_idx",
    "history_len",
    "pred_len",
}


@dataclass(frozen=True)
class LightweightExpertConfig:
    """Stage 1.4a 轻量专家缓存配置。"""

    stage: str = "stage1_4a_lightweight_expert_cache"
    expert_set_id: str = "lightweight_v1"
    recent_mean_fraction: float = 0.25
    soft_oracle_temperature: float = 1.0
    eps: float = 1e-8
    random_seed: int = 20260607


def normalize_expert_ids(expert_ids: Sequence[str] | None = None) -> tuple[str, ...]:
    """校验并规范化本次实际运行的轻量专家集合。"""

    selected = tuple(EXPERT_IDS if expert_ids is None else expert_ids)
    if not selected:
        raise ValueError("expert_ids 不能为空")
    unknown = [expert_id for expert_id in selected if expert_id not in EXPERT_IDS]
    if unknown:
        raise ValueError(f"未知轻量专家：{unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"expert_ids 不能重复：{selected}")
    return selected


def validate_registry(registry: pd.DataFrame) -> None:
    missing = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少必要列：{sorted(missing)}")
    if not registry["physical_window_id"].is_unique:
        raise ValueError("registry 中 physical_window_id 不唯一")
    if (registry["pred_len"].astype(int) <= 0).any():
        raise ValueError("registry 中 pred_len 必须为正整数")
    if (registry["history_len"].astype(int) <= 0).any():
        raise ValueError("registry 中 history_len 必须为正整数")


def _finite_history(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    filled = arr.astype(float, copy=True)
    mean_value = float(np.mean(filled[finite]))
    filled[~finite] = mean_value
    return filled


def _last_value(history: np.ndarray, pred_len: int) -> np.ndarray:
    return np.full(pred_len, float(history[-1]), dtype=float)


def _seasonal_naive(history: np.ndarray, pred_len: int, period: int) -> np.ndarray:
    period = int(period)
    if period <= 0:
        return _last_value(history, pred_len)
    base = history[-period:] if len(history) >= period else history
    repeats = int(math.ceil(pred_len / len(base)))
    return np.tile(base, repeats)[:pred_len].astype(float)


def _recent_mean(history: np.ndarray, pred_len: int, fraction: float) -> np.ndarray:
    width = max(1, int(math.ceil(len(history) * fraction)))
    value = float(np.mean(history[-width:]))
    return np.full(pred_len, value, dtype=float)


def _linear_trend(history: np.ndarray, pred_len: int) -> np.ndarray:
    if len(history) < 2:
        return _last_value(history, pred_len)
    x = np.arange(len(history), dtype=float)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return _last_value(history, pred_len)
    slope = float(np.dot(x_centered, history - float(np.mean(history))) / denom)
    intercept = float(np.mean(history) - slope * np.mean(x))
    future_x = np.arange(len(history), len(history) + pred_len, dtype=float)
    return intercept + slope * future_x


def compute_lightweight_expert_predictions(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    config: LightweightExpertConfig | None = None,
    expert_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """对每个 physical_window_id 只用 history 计算指定轻量专家预测。"""

    validate_registry(registry)
    cfg = config or LightweightExpertConfig()
    selected_expert_ids = normalize_expert_ids(expert_ids)
    rows: list[dict[str, object]] = []

    for row in registry.itertuples(index=False):
        physical_window_id = str(row.physical_window_id)
        if physical_window_id not in histories:
            raise KeyError(f"缺少 history：{physical_window_id}")
        history = _finite_history(histories[physical_window_id])
        pred_len = int(row.pred_len)
        if len(history) != int(row.history_len):
            raise ValueError(f"{physical_window_id} history 长度 {len(history)} != {int(row.history_len)}")

        expert_predictions = {
            "last_value": _last_value(history, pred_len),
            "seasonal_naive": _seasonal_naive(history, pred_len, int(row.period)),
            "recent_mean": _recent_mean(history, pred_len, cfg.recent_mean_fraction),
            "linear_trend": _linear_trend(history, pred_len),
        }

        for expert_id in selected_expert_ids:
            prediction = expert_predictions[expert_id]
            out_row = {
                "physical_window_id": physical_window_id,
                "window_id": str(row.window_id),
                "base_registry_id": str(row.base_registry_id),
                "sample_set_id": str(row.sample_set_id),
                "subset": str(row.subset),
                "split": str(row.split),
                "item_id": str(row.item_id),
                "channel": str(row.channel),
                "period": int(row.period),
                "official_tsf_cell": str(row.official_tsf_cell),
                "history_start_idx": int(row.history_start_idx),
                "target_start_idx": int(row.target_start_idx),
                "pred_len": pred_len,
                "expert_id": expert_id,
                "expert_family": EXPERT_FAMILY[expert_id],
                "prediction_format": "wide_columns",
            }
            for horizon_idx, value in enumerate(prediction):
                out_row[f"yhat_{horizon_idx}"] = float(value)
            rows.append(out_row)

    predictions = pd.DataFrame(rows)
    if predictions[["physical_window_id", "expert_id"]].duplicated().any():
        raise ValueError("predictions 中 (physical_window_id, expert_id) 不唯一")
    return predictions


def _prediction_columns(predictions: pd.DataFrame) -> list[str]:
    cols = [col for col in predictions.columns if col.startswith("yhat_")]
    return sorted(cols, key=lambda name: int(name.split("_", 1)[1]))


def compute_error_table(
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
    config: LightweightExpertConfig | None = None,
) -> pd.DataFrame:
    """用 target 计算误差和 soft oracle；target 不参与专家输入。"""

    cfg = config or LightweightExpertConfig()
    yhat_cols = _prediction_columns(predictions)
    if not yhat_cols:
        raise ValueError("predictions 缺少 yhat_* 预测列")

    rows: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        row_dict = row._asdict()
        physical_window_id = str(row_dict["physical_window_id"])
        if physical_window_id not in targets:
            raise KeyError(f"缺少 target：{physical_window_id}")
        target = np.asarray(targets[physical_window_id], dtype=float)
        yhat = np.asarray([row_dict[col] for col in yhat_cols], dtype=float)
        if len(target) != len(yhat):
            raise ValueError(f"{physical_window_id} target 长度 {len(target)} != prediction 长度 {len(yhat)}")
        diff = yhat - target
        rows.append(
            {
                "physical_window_id": physical_window_id,
                "sample_set_id": str(row_dict["sample_set_id"]),
                "split": str(row_dict["split"]),
                "subset": str(row_dict["subset"]),
                "item_id": str(row_dict["item_id"]),
                "channel": str(row_dict["channel"]),
                "official_tsf_cell": str(row_dict["official_tsf_cell"]),
                "expert_id": str(row_dict["expert_id"]),
                "mse": float(np.mean(diff * diff)),
                "mae": float(np.mean(np.abs(diff))),
            }
        )

    errors = pd.DataFrame(rows)
    errors["rank_in_window"] = errors.groupby("physical_window_id")["mse"].rank(method="first", ascending=True).astype(int)
    errors["is_oracle_top1"] = errors["rank_in_window"] == 1

    errors["soft_oracle_weight"] = 0.0
    for _, group in errors.groupby("physical_window_id", sort=False):
        values = -group["mse"].to_numpy(dtype=float) / max(float(cfg.soft_oracle_temperature), cfg.eps)
        values = values - float(np.max(values))
        weights = np.exp(values)
        weights = weights / max(float(weights.sum()), cfg.eps)
        errors.loc[group.index, "soft_oracle_weight"] = weights
    return errors


def compute_oracle_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """汇总 oracle ensemble、best fixed expert 和 uniform ensemble 的窗口级上界。"""

    if errors.empty:
        return pd.DataFrame(
            [
                {
                    "num_windows": 0,
                    "num_experts": 0,
                    "oracle_mse": np.nan,
                    "best_fixed_expert": "",
                    "best_fixed_mse": np.nan,
                    "uniform_ensemble_mse_proxy": np.nan,
                    "oracle_gap_vs_best_fixed": np.nan,
                }
            ]
        )

    oracle_mse = float(errors.groupby("physical_window_id")["mse"].min().mean())
    fixed = errors.groupby("expert_id")["mse"].mean().sort_values()
    best_fixed_expert = str(fixed.index[0])
    best_fixed_mse = float(fixed.iloc[0])
    uniform_proxy = float(errors.groupby("physical_window_id")["mse"].mean().mean())
    return pd.DataFrame(
        [
            {
                "num_windows": int(errors["physical_window_id"].nunique()),
                "num_experts": int(errors["expert_id"].nunique()),
                "oracle_mse": oracle_mse,
                "best_fixed_expert": best_fixed_expert,
                "best_fixed_mse": best_fixed_mse,
                "uniform_ensemble_mse_proxy": uniform_proxy,
                "oracle_gap_vs_best_fixed": best_fixed_mse - oracle_mse,
            }
        ]
    )


def select_stratified_registry(
    registry: pd.DataFrame,
    target_rows: int,
    stratify_columns: Sequence[str],
    random_seed: int,
) -> pd.DataFrame:
    """按给定列做确定性分层抽样，尽量让每个非空 cell 均衡覆盖。"""

    if target_rows <= 0:
        raise ValueError("target_rows 必须为正整数")
    missing = set(stratify_columns) - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少分层列：{sorted(missing)}")
    if target_rows >= len(registry):
        return registry.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    group_keys = list(stratify_columns)
    groups: list[pd.DataFrame] = []
    for _, group in registry.groupby(group_keys, sort=True, dropna=False):
        # 先在组内随机打散，再由全局 quota 控制每组取多少，保证可复现且不依赖原始排序。
        order = rng.permutation(len(group))
        groups.append(group.iloc[order].reset_index(drop=True))

    base_quota = target_rows // len(groups)
    remainder = target_rows % len(groups)
    allocations = [min(len(group), base_quota) for group in groups]
    remaining = target_rows - sum(allocations)

    cursor = 0
    while remaining > 0:
        group_idx = cursor % len(groups)
        if allocations[group_idx] < len(groups[group_idx]):
            allocations[group_idx] += 1
            remaining -= 1
        cursor += 1
        if cursor > len(groups) * (target_rows + 1):
            raise RuntimeError("分层抽样 quota 分配未能收敛")

    sampled = pd.concat([group.iloc[:take] for group, take in zip(groups, allocations) if take > 0], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    validate_registry(sampled)
    return sampled


def build_cell_model_matrix(errors: pd.DataFrame) -> pd.DataFrame:
    """按 official TSF cell 汇总每个 expert 的平均误差和胜率。"""

    grouped = (
        errors.groupby(["official_tsf_cell", "expert_id"], as_index=False)
        .agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            oracle_top1_rate=("is_oracle_top1", "mean"),
            num_windows=("physical_window_id", "nunique"),
        )
        .sort_values(["official_tsf_cell", "mse", "expert_id"])
        .reset_index(drop=True)
    )
    grouped["rank_in_cell"] = grouped.groupby("official_tsf_cell")["mse"].rank(method="first", ascending=True).astype(int)
    return grouped


def build_cache_manifest(
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    elapsed_seconds: float,
    input_registry_dir: Path,
    max_rows: int | None,
    stratified_rows: int | None,
    stratify_columns: Sequence[str],
    expert_ids: Sequence[str] | None = None,
    config: LightweightExpertConfig | None = None,
) -> dict[str, object]:
    cfg = config or LightweightExpertConfig()
    selected_expert_ids = normalize_expert_ids(expert_ids)
    actual_prediction_experts = tuple(sorted(predictions["expert_id"].astype(str).unique().tolist()))
    actual_error_experts = tuple(sorted(errors["expert_id"].astype(str).unique().tolist()))
    expected_experts = tuple(sorted(selected_expert_ids))
    if actual_prediction_experts != expected_experts or actual_error_experts != expected_experts:
        raise ValueError("expert_ids 与 predictions/errors 实际专家集合不一致")
    return {
        "stage": cfg.stage,
        "expert_set_id": cfg.expert_set_id,
        "expert_ids": list(selected_expert_ids),
        "expert_families": {expert_id: EXPERT_FAMILY[expert_id] for expert_id in selected_expert_ids},
        "sample_set_id": sorted(registry["sample_set_id"].astype(str).unique().tolist()),
        "base_registry_id": sorted(registry["base_registry_id"].astype(str).unique().tolist()),
        "input_registry_dir": str(input_registry_dir),
        "max_rows": max_rows,
        "stratified_rows": stratified_rows,
        "stratify_columns": list(stratify_columns),
        "total_windows": int(registry["physical_window_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "error_rows": int(len(errors)),
        "unique_prediction_key": bool(not predictions[["physical_window_id", "expert_id"]].duplicated().any()),
        "unique_error_key": bool(not errors[["physical_window_id", "expert_id"]].duplicated().any()),
        "split_window_counts": registry.groupby("split")["physical_window_id"].nunique().to_dict(),
        "subset_window_counts": registry.groupby("subset")["physical_window_id"].nunique().to_dict(),
        "cell_window_counts": registry.groupby("official_tsf_cell")["physical_window_id"].nunique().to_dict(),
        "prediction_format": "wide_columns",
        "future_read_policy": "history_only_for_prediction",
        "target_usage": "error_and_oracle_only",
        "implements_router": False,
        "runs_visual_encoder": False,
        "runs_neural_experts": False,
        "modifies_quito_code": False,
        "elapsed_seconds": float(elapsed_seconds),
        "latency_ms_per_window": float(elapsed_seconds * 1000.0 / max(len(registry), 1)),
        "config": asdict(cfg),
        "output_files": {
            "predictions": "predictions.parquet",
            "errors": "errors.parquet",
            "manifest": "manifest.json",
            "cell_model_matrix": "profiling/cell_model_matrix.csv",
            "oracle_summary": "profiling/oracle_summary.csv",
        },
    }


def write_expert_cache_outputs(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    cell_model_matrix: pd.DataFrame,
    manifest: Mapping[str, object],
    output_root: Path,
    expert_set_id: str,
) -> Path:
    out_dir = output_root / expert_set_id
    profiling_dir = out_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    errors.to_parquet(out_dir / "errors.parquet", index=False)
    cell_model_matrix.to_csv(profiling_dir / "cell_model_matrix.csv", index=False)
    oracle_summary.to_csv(profiling_dir / "oracle_summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def load_registry(
    registry_dir: Path,
    max_rows: int | None = None,
    stratified_rows: int | None = None,
    stratify_columns: Sequence[str] = ("split", "subset", "official_tsf_cell"),
    random_seed: int = 20260607,
) -> tuple[pd.DataFrame, dict[str, object]]:
    registry_path = registry_dir / "window_index.csv"
    manifest_path = registry_dir / "manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"registry 不存在：{registry_path}")
    registry = pd.read_csv(registry_path)
    if max_rows is not None and stratified_rows is not None:
        raise ValueError("--max-rows 和 --stratified-rows 不能同时使用")
    if stratified_rows is not None:
        registry = select_stratified_registry(
            registry,
            target_rows=stratified_rows,
            stratify_columns=stratify_columns,
            random_seed=random_seed,
        )
    elif max_rows is not None:
        registry = registry.head(max_rows).copy()
    validate_registry(registry)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return registry, manifest


def extract_histories_and_targets(
    registry: pd.DataFrame,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按 registry 边界从 QuitoBench 数据中抽取 history 和 target。"""

    subsets = tuple(sorted(registry["subset"].astype(str).unique().tolist()))
    frames = load_subset_frames(data_dir, subsets)
    histories: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    frame_cache: dict[tuple[str, int], pd.DataFrame] = {}
    source_cache: dict[tuple[str, int, str], np.ndarray] = {}
    for row in registry.itertuples(index=False):
        subset = str(row.subset)
        item_id = int(row.item_id)
        channel = str(row.channel)
        frame_key = (subset, item_id)
        if frame_key not in frame_cache:
            if subset not in frames:
                raise ValueError(f"缺少 subset 原始数据：{subset}")
            item_frame = frames[subset][frames[subset]["item_id"] == item_id]
            if item_frame.empty:
                raise ValueError(f"原始数据缺少 {subset}/{item_id}")
            frame_cache[frame_key] = item_frame.sort_values("date_time").reset_index(drop=True)
        source_key = (subset, item_id, channel)
        if source_key not in source_cache:
            ordered = frame_cache[frame_key]
            if channel not in ordered.columns:
                raise ValueError(f"原始数据缺少 channel：{subset}/{item_id}/{channel}")
            source_cache[source_key] = ordered[channel].to_numpy(dtype=float)
        values = source_cache[source_key]
        history = values[int(row.history_start_idx) : int(row.history_end_idx)]
        target = values[int(row.target_start_idx) : int(row.target_end_idx)]
        if len(history) != int(row.history_len):
            raise ValueError(f"{row.physical_window_id} history 长度 {len(history)} != {int(row.history_len)}")
        if len(target) != int(row.pred_len):
            raise ValueError(f"{row.physical_window_id} target 长度 {len(target)} != {int(row.pred_len)}")
        histories[str(row.physical_window_id)] = history
        targets[str(row.physical_window_id)] = target
    return histories, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expert-set-id", default="lightweight_v1")
    parser.add_argument(
        "--expert-ids",
        default=",".join(EXPERT_IDS),
        help="逗号分隔的轻量专家列表；全量 seasonal baseline 可设为 seasonal_naive",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--stratified-rows", type=int, default=None)
    parser.add_argument(
        "--stratify-columns",
        default="split,subset,official_tsf_cell",
        help="逗号分隔的分层列，默认 split,subset,official_tsf_cell",
    )
    parser.add_argument("--recent-mean-fraction", type=float, default=0.25)
    parser.add_argument("--soft-oracle-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LightweightExpertConfig(
        expert_set_id=args.expert_set_id,
        recent_mean_fraction=args.recent_mean_fraction,
        soft_oracle_temperature=args.soft_oracle_temperature,
    )
    start = time.perf_counter()
    expert_ids = normalize_expert_ids(tuple(expert_id.strip() for expert_id in args.expert_ids.split(",") if expert_id.strip()))
    stratify_columns = tuple(col.strip() for col in args.stratify_columns.split(",") if col.strip())
    registry, registry_manifest = load_registry(
        args.registry_dir,
        max_rows=args.max_rows,
        stratified_rows=args.stratified_rows,
        stratify_columns=stratify_columns,
        random_seed=config.random_seed,
    )
    histories, targets = extract_histories_and_targets(registry, data_dir=args.data_dir)
    predictions = compute_lightweight_expert_predictions(registry, histories, config=config, expert_ids=expert_ids)
    errors = compute_error_table(predictions, targets, config=config)
    oracle_summary = compute_oracle_summary(errors)
    cell_model_matrix = build_cell_model_matrix(errors)
    elapsed = time.perf_counter() - start
    manifest = build_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=elapsed,
        input_registry_dir=args.registry_dir,
        max_rows=args.max_rows,
        stratified_rows=args.stratified_rows,
        stratify_columns=stratify_columns,
        expert_ids=expert_ids,
        config=config,
    )
    manifest["input_registry_manifest"] = registry_manifest
    out_dir = write_expert_cache_outputs(
        predictions=predictions,
        errors=errors,
        oracle_summary=oracle_summary,
        cell_model_matrix=cell_model_matrix,
        manifest=manifest,
        output_root=args.output_root,
        expert_set_id=args.expert_set_id,
    )
    print(f"[done] output_dir={out_dir}")
    print(f"[done] windows={manifest['total_windows']}")
    print(f"[done] prediction_rows={manifest['prediction_rows']}")
    print(f"[done] latency_ms_per_window={manifest['latency_ms_per_window']:.4f}")


if __name__ == "__main__":
    main()
