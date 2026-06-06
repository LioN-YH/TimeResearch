"""Stage 1.1：QuitoBench sample-channel 轻量 proxy 预计算。

本脚本读取 Stage 1.0 working registry，只使用每个 sample-channel 的
history 窗口计算在线可复现的轻量统计特征。禁止读取 future target，
不做 full STL，不运行专家模型，也不实现 router。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_window_registry import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_ROOT, load_subset_frames


DEFAULT_REGISTRY_DIR = (
    DEFAULT_OUTPUT_ROOT / "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e"
)
DEFAULT_PROXY_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/proxy_features"

ID_COLUMNS = [
    "physical_window_id",
    "window_id",
    "base_registry_id",
    "sample_set_id",
    "subset",
    "split",
    "item_id",
    "channel",
    "period",
    "history_start_idx",
    "history_end_idx",
    "target_start_idx",
    "target_end_idx",
    "history_len",
    "pred_len",
]
OPTIONAL_CONTEXT_COLUMNS = ["official_cluster", "official_tsf_cell"]
FEATURE_COLUMNS = [
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "amplitude",
    "last_value",
    "missing_ratio",
    "slope",
    "recent_std_ratio",
    "acf_lag1",
    "acf_period",
    "spectral_entropy",
    "dominant_frequency_strength",
]


@dataclass(frozen=True)
class ProxyConfig:
    """轻量 proxy 配置。

    recent_fraction 用于 recent volatility proxy，默认取 history 尾部 25%。
    fft_eps 用于避免频域归一化时除零。
    """

    stage: str = "stage1_1_sample_channel_light_proxy"
    feature_set: str = "light_v1_compact"
    recent_fraction: float = 0.25
    fft_eps: float = 1e-12


def _finite_values(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    return arr, finite_mask


def _fill_nonfinite_with_mean(values: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    if not finite_mask.any():
        return np.zeros_like(values, dtype=float)
    filled = values.astype(float, copy=True)
    mean_value = float(np.mean(filled[finite_mask]))
    filled[~finite_mask] = mean_value
    return filled


def _safe_autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return 0.0
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def _safe_slope(values: np.ndarray, finite_mask: np.ndarray) -> float:
    if finite_mask.sum() < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)[finite_mask]
    y = values[finite_mask]
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y - float(np.mean(y))) / denom)


def _spectral_features(values: np.ndarray, eps: float) -> tuple[float, float]:
    if len(values) < 4:
        return 0.0, 0.0
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 2:
        return 0.0, 0.0
    nonzero_power = power[1:]
    total = float(np.sum(nonzero_power))
    if total <= eps:
        return 0.0, 0.0
    probs = nonzero_power / total
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    normalized_entropy = entropy / math.log(len(probs))
    dominant_strength = float(np.max(nonzero_power) / total)
    return float(np.clip(normalized_entropy, 0.0, 1.0)), float(np.clip(dominant_strength, 0.0, 1.0))


def compute_window_proxy(values: Sequence[float], period: int, config: ProxyConfig | None = None) -> dict[str, float]:
    """只基于 history values 计算单个窗口的轻量 proxy。

    调用方必须传入已经切好的 history 序列；本函数没有 target/future 参数，
    从接口层面降低误读 future 的风险。
    """

    cfg = config or ProxyConfig()
    arr, finite_mask = _finite_values(values)
    if len(arr) == 0:
        raise ValueError("history 窗口不能为空")
    missing_ratio = float(1.0 - finite_mask.mean())
    if finite_mask.any():
        clean = arr[finite_mask]
        mean_value = float(np.mean(clean))
        std_value = float(np.std(clean))
        median_value = float(np.median(clean))
        q25, q75 = np.quantile(clean, [0.25, 0.75])
        min_value = float(np.min(clean))
        max_value = float(np.max(clean))
        last_candidates = arr[finite_mask]
        last_value = float(last_candidates[-1])
    else:
        mean_value = std_value = median_value = 0.0
        q25 = q75 = 0.0
        min_value = max_value = last_value = 0.0

    filled = _fill_nonfinite_with_mean(arr, finite_mask)
    recent_len = max(2, int(math.ceil(len(filled) * cfg.recent_fraction)))
    recent_std = float(np.std(filled[-recent_len:]))
    full_std = float(np.std(filled))
    spectral_entropy, dominant_frequency_strength = _spectral_features(filled, cfg.fft_eps)

    return {
        "mean": mean_value,
        "std": std_value,
        "median": median_value,
        "iqr": float(q75 - q25),
        "min": min_value,
        "max": max_value,
        "amplitude": float(max_value - min_value),
        "last_value": last_value,
        "missing_ratio": missing_ratio,
        "slope": _safe_slope(arr, finite_mask),
        "recent_std_ratio": float(recent_std / (full_std + 1e-8)),
        "acf_lag1": _safe_autocorr(filled, 1),
        "acf_period": _safe_autocorr(filled, int(period)),
        "spectral_entropy": spectral_entropy,
        "dominant_frequency_strength": dominant_frequency_strength,
    }


def _validate_registry(registry: pd.DataFrame) -> None:
    missing = set(ID_COLUMNS) - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少必要列：{sorted(missing)}")
    if not registry["physical_window_id"].is_unique:
        raise ValueError("registry 中 physical_window_id 不唯一")


def compute_light_proxy_features(
    registry: pd.DataFrame,
    subset_frames: Mapping[str, pd.DataFrame],
    config: ProxyConfig,
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """为 registry 中每个 sample-channel 窗口计算轻量 proxy。"""

    _validate_registry(registry)
    rows: list[dict[str, object]] = []
    source_cache: dict[tuple[str, int, str], np.ndarray] = {}
    frame_cache: dict[tuple[str, int], pd.DataFrame] = {}
    context_columns = ID_COLUMNS + [c for c in OPTIONAL_CONTEXT_COLUMNS if c in registry.columns]

    for idx, row in enumerate(registry.itertuples(index=False), start=1):
        row_data = row._asdict()
        subset = str(row_data["subset"])
        item_id = int(row_data["item_id"])
        channel = str(row_data["channel"])
        frame_key = (subset, item_id)
        if frame_key not in frame_cache:
            if subset not in subset_frames:
                raise ValueError(f"缺少 subset 原始数据：{subset}")
            item_df = subset_frames[subset][subset_frames[subset]["item_id"] == item_id]
            if item_df.empty:
                raise ValueError(f"原始数据缺少 {subset}/{item_id}")
            frame_cache[frame_key] = item_df.sort_values("date_time").reset_index(drop=True)
        source_key = (subset, item_id, channel)
        if source_key not in source_cache:
            item_df = frame_cache[frame_key]
            if channel not in item_df.columns:
                raise ValueError(f"原始数据缺少 channel：{subset}/{item_id}/{channel}")
            source_cache[source_key] = item_df[channel].to_numpy(dtype=float)

        values = source_cache[source_key]
        history_start = int(row_data["history_start_idx"])
        history_end = int(row_data["history_end_idx"])
        if history_start < 0 or history_end > len(values) or history_start >= history_end:
            raise ValueError(
                f"非法 history 边界：{row_data['physical_window_id']} "
                f"[{history_start}, {history_end}) len={len(values)}"
            )
        history_values = values[history_start:history_end]
        out_row = {col: row_data[col] for col in context_columns}
        out_row.update(compute_window_proxy(history_values, period=int(row_data["period"]), config=config))
        rows.append(out_row)
        if progress_every and idx % progress_every == 0:
            print(f"[progress] proxy rows={idx}/{len(registry)}")

    proxy = pd.DataFrame(rows)
    if not proxy.empty:
        proxy = proxy.sort_values(["subset", "item_id", "split", "target_start_idx", "channel"]).reset_index(drop=True)
    sample_set_values = sorted(proxy["sample_set_id"].dropna().unique().tolist()) if not proxy.empty else []
    base_registry_values = sorted(proxy["base_registry_id"].dropna().unique().tolist()) if not proxy.empty else []
    manifest = {
        "stage": config.stage,
        "feature_set": config.feature_set,
        "config": asdict(config),
        "total_windows": int(len(proxy)),
        "sample_set_id": sample_set_values[0] if len(sample_set_values) == 1 else sample_set_values,
        "base_registry_id": base_registry_values[0] if len(base_registry_values) == 1 else base_registry_values,
        "id_columns": context_columns,
        "feature_columns": FEATURE_COLUMNS,
        "split_window_counts": proxy["split"].value_counts().sort_index().astype(int).to_dict() if not proxy.empty else {},
        "subset_window_counts": proxy["subset"].value_counts().sort_index().astype(int).to_dict() if not proxy.empty else {},
        "unique_physical_window_id": bool(proxy["physical_window_id"].is_unique) if not proxy.empty else True,
        "future_read_policy": "history_only",
        "uses_full_stl": False,
        "runs_expert_models": False,
        "implements_router": False,
    }
    return proxy, manifest


def load_registry(registry_dir: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取 Stage 1.0 registry CSV 和 manifest。"""

    registry_path = registry_dir / "window_index.csv"
    manifest_path = registry_dir / "manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"缺少 registry：{registry_path}")
    registry = pd.read_csv(registry_path, nrows=max_rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return registry, manifest


def write_proxy_outputs(
    proxy: pd.DataFrame,
    manifest: Mapping[str, object],
    output_root: Path,
    output_format: str = "auto",
    run_scope: str = "full",
    max_rows: int | None = None,
) -> Path:
    """写出 proxy feature table 和 manifest。"""

    sample_set_values = sorted(proxy["sample_set_id"].dropna().unique().tolist()) if not proxy.empty else []
    if len(sample_set_values) != 1:
        raise ValueError(f"proxy 输出必须对应唯一 sample_set_id，当前为：{sample_set_values}")
    sample_set_id = str(sample_set_values[0])
    out_name = sample_set_id
    if run_scope != "full" or max_rows is not None:
        suffix = run_scope
        if max_rows is not None:
            suffix = f"{suffix}_max_rows_{int(max_rows)}"
        out_name = f"{sample_set_id}__{suffix}"
    out_dir = output_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_format = output_format
    if output_format == "auto":
        actual_format = "parquet"
    manifest_to_write = dict(manifest)
    try:
        if actual_format == "parquet":
            output_file = out_dir / "sample_channel_proxy.parquet"
            proxy.to_parquet(output_file, index=False)
        elif actual_format == "csv":
            output_file = out_dir / "sample_channel_proxy.csv"
            proxy.to_csv(output_file, index=False)
        else:
            raise ValueError(f"未知输出格式：{output_format}")
    except ImportError:
        output_file = out_dir / "sample_channel_proxy.csv"
        proxy.to_csv(output_file, index=False)
        actual_format = "csv"

    manifest_to_write["run_scope"] = run_scope
    manifest_to_write["max_rows"] = max_rows
    manifest_to_write["output_dir_name"] = out_name
    manifest_to_write["output_file"] = output_file.name
    manifest_to_write["output_format"] = actual_format
    (out_dir / "manifest.json").write_text(json.dumps(manifest_to_write, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROXY_OUTPUT_ROOT)
    parser.add_argument("--output-format", choices=["auto", "parquet", "csv"], default="auto")
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于 smoke/debug，限制读取 registry 前 N 行")
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProxyConfig()
    registry, registry_manifest = load_registry(args.registry_dir, max_rows=args.max_rows)
    subsets = tuple(sorted(registry["subset"].dropna().unique().tolist()))
    print(f"[input] registry_dir={args.registry_dir} rows={len(registry)} subsets={subsets}")
    frames = load_subset_frames(args.data_dir, subsets)
    proxy, manifest = compute_light_proxy_features(
        registry=registry,
        subset_frames=frames,
        config=config,
        progress_every=args.progress_every,
    )
    manifest["input_registry_dir"] = str(args.registry_dir)
    manifest["input_registry_manifest"] = registry_manifest
    run_scope = "smoke" if args.max_rows is not None else "full"
    out_dir = write_proxy_outputs(
        proxy,
        manifest,
        args.output_root,
        output_format=args.output_format,
        run_scope=run_scope,
        max_rows=args.max_rows,
    )
    print(f"[done] proxy_rows={len(proxy)} output={out_dir}")
    print(f"[done] split_window_counts={manifest['split_window_counts']}")
    print(f"[done] subset_window_counts={manifest['subset_window_counts']}")


if __name__ == "__main__":
    main()
