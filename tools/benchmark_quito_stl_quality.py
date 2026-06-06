"""Quito/STL 质量指标耗时基准。

本脚本只做少量代表序列计时，用于估算全量 Quito `evaluate_series`
在 item 级和通道级质量审计中的运行成本。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))

from quito.utils.dataset_quality import evaluate_series  # noqa: E402


def representative_series(item_df: pd.DataFrame) -> np.ndarray:
    """把单个 item 的 L x C 矩阵转为 item 级代表序列。

    输入 shape：
    - item_df: 包含一个 item 的长表，指标列为 ind_1..ind_5。

    输出 shape：
    - representative: L，一维数组。

    说明：
    - 仅用于 CPU 质量指标计时。
    - 先对每个通道 z-score，再沿通道求均值，避免量纲支配。
    """

    cols = [c for c in item_df.columns if c.startswith("ind_")]
    values = item_df.sort_values("date_time")[cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True) + 1e-8
    return np.nanmean((values - mean) / std, axis=1)


def pick_cases() -> list[tuple[str, str, int, np.ndarray]]:
    """挑选少量代表序列，覆盖 hour/min、item/channel、全长/降采样。"""

    cases: list[tuple[str, str, int, np.ndarray]] = []
    for subset, period in [("hour", 24), ("min", 144)]:
        path = ROOT / "data/hf/hq-bench/quitobench/v20260315" / f"test_{subset}-00001-of-00001.parquet"
        df = pd.read_parquet(path)
        first_id = df["item_id"].iloc[0]
        item_df = df[df["item_id"] == first_id].sort_values("date_time")
        item_x = representative_series(item_df)
        ch_x = item_df["ind_1"].to_numpy(dtype=float)
        idx = np.linspace(0, len(item_x) - 1, 2048, dtype=int)
        cases.append((subset, "item_mean_z_full", period, item_x))
        cases.append((subset, "single_channel_full", period, ch_x))
        cases.append((subset, "item_mean_z_2048", period, item_x[idx]))
    return cases


def main() -> None:
    for subset, name, period, values in pick_cases():
        stl_period = min(period, max(2, len(values) // 2))
        t0 = time.perf_counter()
        result = evaluate_series(values, period=stl_period, compute_adf=False)
        seconds = time.perf_counter() - t0
        print(
            f"{subset},{name},len={len(values)},period={stl_period},"
            f"seconds={seconds:.3f},forecastability={result.forecastability:.4f},"
            f"seasonality={result.season_strength:.4f},trend={result.trend_strength:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
