"""QuitoBench 数据充分性审计脚本。

本脚本只读取 QuitoBench benchmark parquet，不读取 Quito 预训练 corpus。
输出用于判断仅使用 QuitoBench 是否足够支撑路线 1 和路线 2。

统计口径：
- item 级：每个 item_id 是一条多变量序列，质量指标按 5 个指标列的均值聚合。
- 通道级：每个 (item_id, ind_k) 是一条单变量序列，匹配后续默认“通道独立”伪图像策略。
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))


CUTOFF = pd.Timestamp("2023-07-28 00:00:00")
WINDOW_SETTINGS = [(96, 96), (192, 96), (336, 96)]


@dataclass(frozen=True)
class SeriesTask:
    """单条质量指标计算任务。

    输入 shape：
    - values: 一维 numpy array，长度为当前 item/channel 的时间步数。

    输出：
    - forecastability、seasonality、trend、missing_ratio、effective_length。

    CPU/GPU 说明：
    - 质量分析全部在 CPU 上执行，不发生 GPU 数据迁移。
    """

    subset: str
    item_id: int
    channel: str
    values: np.ndarray
    period: int
    quality_max_points: int
    quality_method: str


def window_count(length: int, seq_len: int, pred_len: int) -> int:
    """计算单条序列在给定窗口设置下可切出的滑动窗口数。"""

    return max(int(length) - int(seq_len) - int(pred_len) + 1, 0)


def downsample_for_quality(values: np.ndarray, max_points: int) -> np.ndarray:
    """为质量指标计算做可复现等距降采样。

    输入输出 shape：
    - 输入 values: L
    - 输出 values_ds: min(L, max_points)

    说明：
    - 仅用于 STL / 频域质量指标加速。
    - 不影响原始长度、train/valid/test 长度和滑动窗口数量统计。
    """

    if max_points <= 0 or len(values) <= max_points:
        return values
    idx = np.linspace(0, len(values) - 1, num=max_points, dtype=int)
    return values[idx]


def clean_1d(values: np.ndarray) -> tuple[np.ndarray, int, float]:
    """清理一维序列并返回缺失统计。

    输出：
    - clean: shape 为 L 的 float array，NaN 用均值填充。
    - effective_length: 非 NaN 点数。
    - missing_ratio: NaN 占比。
    """

    x = np.asarray(values, dtype=float).reshape(-1)
    missing = np.isnan(x)
    effective_length = int((~missing).sum())
    missing_ratio = float(missing.mean()) if len(x) else 1.0
    if effective_length == 0:
        return np.zeros_like(x, dtype=float), effective_length, missing_ratio
    fill = float(np.nanmean(x))
    return np.nan_to_num(x, nan=fill), effective_length, missing_ratio


def fft_forecastability(values: np.ndarray) -> float:
    """用 FFT 频谱熵近似 forecastability，返回 [0, 1]。"""

    x, _, _ = clean_1d(values)
    if len(x) == 0 or np.nanstd(x) < 1e-12:
        return 1.0
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(power) > 1:
        power = power[1:]
    power = power + 1e-12
    prob = power / power.sum()
    entropy = float(-(prob * np.log(prob)).sum())
    entropy_max = float(np.log(len(prob))) if len(prob) > 1 else 1.0
    return float(np.clip(1.0 - entropy / entropy_max, 0.0, 1.0))


def autocorr_seasonality(values: np.ndarray, period: int) -> float:
    """用日周期滞后自相关近似 seasonality strength，返回 [0, 1]。"""

    x, _, _ = clean_1d(values)
    if len(x) <= period or np.std(x) < 1e-12:
        return 0.0
    a = x[:-period]
    b = x[period:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.clip(abs(np.corrcoef(a, b)[0, 1]), 0.0, 1.0))


def linear_trend_strength(values: np.ndarray) -> float:
    """用线性趋势解释方差 R2 近似 trend strength，返回 [0, 1]。"""

    x, _, _ = clean_1d(values)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0
    t = np.linspace(0.0, 1.0, num=len(x))
    slope, intercept = np.polyfit(t, x, deg=1)
    pred = slope * t + intercept
    sse = float(np.sum((x - pred) ** 2))
    sst = float(np.sum((x - np.mean(x)) ** 2)) + 1e-12
    return float(np.clip(1.0 - sse / sst, 0.0, 1.0))


def quantile_summary(values: pd.Series) -> dict[str, float]:
    """生成分布摘要，用于报告长度和质量指标的主要分位数。"""

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"min": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "max": math.nan}
    qs = clean.quantile([0.25, 0.5, 0.75])
    return {
        "min": float(clean.min()),
        "p25": float(qs.loc[0.25]),
        "p50": float(qs.loc[0.5]),
        "p75": float(qs.loc[0.75]),
        "max": float(clean.max()),
    }


def split_lengths_for_item(item_df: pd.DataFrame) -> tuple[int, int, int]:
    """按 QuitoBench cutoff 重建 train/valid/test 长度。

    cutoff 前的数据再按 Quito `DatasetConfig.split(test=False)` 的逻辑划分：
    valid_size = int(pre_cutoff_len * 0.2)，train_size = pre_cutoff_len - valid_size。
    test 使用 cutoff 及之后的数据长度。
    """

    dates = pd.to_datetime(item_df["date_time"])
    pre_cutoff_len = int((dates < CUTOFF).sum())
    test_len = int((dates >= CUTOFF).sum())
    valid_len = int(pre_cutoff_len * 0.2)
    train_len = pre_cutoff_len - valid_len
    return train_len, valid_len, test_len


def make_quality_task(task: SeriesTask) -> dict[str, object]:
    """计算单条序列质量指标。

    关键超参数：
    - period: STL 周期。hour 使用 24，min 使用 144，对应一天周期。
    - quality_method: light 使用轻量近似；stl 使用 Quito 原生 STL 指标，较慢。
    """

    values = downsample_for_quality(task.values.astype(float), task.quality_max_points)
    period = min(task.period, max(2, len(values) // 2))
    clean, effective_length, missing_ratio = clean_1d(values)
    if task.quality_method == "stl":
        from quito.utils.dataset_quality import evaluate_series

        result = evaluate_series(clean, period=period, compute_adf=False)
        forecastability = result.forecastability
        seasonality_strength = result.season_strength
        trend_strength = result.trend_strength
    else:
        forecastability = fft_forecastability(clean)
        seasonality_strength = autocorr_seasonality(clean, period)
        trend_strength = linear_trend_strength(clean)
    return {
        "subset": task.subset,
        "item_id": task.item_id,
        "channel": task.channel,
        "quality_method": task.quality_method,
        "quality_points": len(values),
        "effective_length": effective_length,
        "missing_ratio": missing_ratio,
        "forecastability": forecastability,
        "seasonality_strength": seasonality_strength,
        "trend_strength": trend_strength,
    }


def iter_quality_tasks(
    subset: str, df: pd.DataFrame, indicator_cols: list[str], period: int, quality_max_points: int, quality_method: str
) -> Iterable[SeriesTask]:
    """生成通道级质量任务，默认匹配通道独立策略。"""

    for item_id, item_df in df.groupby("item_id", sort=True):
        item_df = item_df.sort_values("date_time")
        for channel in indicator_cols:
            yield SeriesTask(
                subset=subset,
                item_id=int(item_id),
                channel=channel,
                values=item_df[channel].to_numpy(),
                period=period,
                quality_max_points=quality_max_points,
                quality_method=quality_method,
            )


def iter_item_quality_tasks(
    subset: str, df: pd.DataFrame, indicator_cols: list[str], period: int, quality_max_points: int, quality_method: str
) -> Iterable[SeriesTask]:
    """生成 item 级质量任务。

    输入 shape：
    - 每个 item 原始矩阵为 L x C，其中 C=5。

    通道独立策略说明：
    - 当前函数不混合作为 router 输入的通道样本，只为 TSF cell 审计构造 item 级代表序列。
    - 代表序列先对每个通道在 CPU 上做 z-score，再沿通道取均值，避免某个指标列量纲支配质量指标。
    """

    for item_id, item_df in df.groupby("item_id", sort=True):
        item_df = item_df.sort_values("date_time")
        values = item_df[indicator_cols].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True) + 1e-8
        z_values = (values - mean) / std
        representative = np.nanmean(z_values, axis=1)
        yield SeriesTask(
            subset=subset,
            item_id=int(item_id),
            channel="item_mean_z",
            values=representative,
            period=period,
            quality_max_points=quality_max_points,
            quality_method=quality_method,
        )


def compute_quality(tasks: list[SeriesTask], max_workers: int) -> pd.DataFrame:
    """并行计算通道级质量指标，返回每行一个 (item_id, channel)。"""

    if max_workers <= 1:
        return pd.DataFrame([make_quality_task(task) for task in tasks])

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(make_quality_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_cells(quality_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """用全局中位数阈值构造 2x2x2 TSF cell。

    输入 shape：
    - quality_df: 行为 channel 或 item，列包含 trend/seasonality/forecastability。

    输出：
    - 增加 trend_bin、seasonality_bin、forecastability_bin、tsf_cell 的 DataFrame。
    - 三个阈值字典。
    """

    df = quality_df.copy()
    thresholds = {
        "trend_threshold": float(df["trend_strength"].median()),
        "seasonality_threshold": float(df["seasonality_strength"].median()),
        "forecastability_threshold": float(df["forecastability"].median()),
    }
    df["trend_bin"] = np.where(df["trend_strength"] >= thresholds["trend_threshold"], "highT", "lowT")
    df["seasonality_bin"] = np.where(
        df["seasonality_strength"] >= thresholds["seasonality_threshold"], "highS", "lowS"
    )
    df["forecastability_bin"] = np.where(df["forecastability"] >= thresholds["forecastability_threshold"], "highF", "lowF")
    df["tsf_cell"] = df["trend_bin"] + "_" + df["seasonality_bin"] + "_" + df["forecastability_bin"]
    return df, thresholds


def item_level_quality(channel_quality: pd.DataFrame) -> pd.DataFrame:
    """把通道级质量指标聚合到 item 级，便于和 QuitoBench 官方“series”口径对齐。"""

    metric_cols = ["effective_length", "missing_ratio", "forecastability", "seasonality_strength", "trend_strength"]
    return (
        channel_quality.groupby(["subset", "item_id"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .assign(channel="item_mean")
    )


def replicate_item_quality_to_channels(item_quality: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    """把 item 级 TSF cell 复制到通道级样本口径。

    这一步不声称逐通道质量指标不同，只用于回答默认通道独立策略下每个 cell 有多少通道样本和窗口。
    """

    rows = []
    for _, row in item_quality.iterrows():
        base = row.to_dict()
        for channel in indicator_cols:
            copied = dict(base)
            copied["channel"] = channel
            rows.append(copied)
    return pd.DataFrame(rows)


def audit_subset(
    subset: str, path: Path, max_workers: int, quality_scope: str, quality_max_points: int, quality_method: str
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """审计单个 QuitoBench subset，并返回后续输出所需表。"""

    df = pd.read_parquet(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    indicator_cols = [c for c in df.columns if c.startswith("ind_")]
    period = 24 if subset == "hour" else 144

    item_rows = []
    window_rows = []
    for item_id, item_df in df.groupby("item_id", sort=True):
        item_df = item_df.sort_values("date_time")
        train_len, valid_len, test_len = split_lengths_for_item(item_df)
        total_len = len(item_df)
        item_rows.append(
            {
                "subset": subset,
                "item_id": int(item_id),
                "total_length": total_len,
                "train_length": train_len,
                "valid_length": valid_len,
                "test_length": test_len,
                "start_time": item_df["date_time"].min(),
                "end_time": item_df["date_time"].max(),
            }
        )
        for seq_len, pred_len in WINDOW_SETTINGS:
            for split_name, split_len in [("train", train_len), ("valid", valid_len), ("test", test_len)]:
                # item 级窗口用于 M 特征模式；通道级窗口乘以指标列数，用于 S/通道独立口径。
                item_windows = window_count(split_len, seq_len, pred_len)
                window_rows.append(
                    {
                        "subset": subset,
                        "item_id": int(item_id),
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "split": split_name,
                        "split_length": split_len,
                        "item_windows": item_windows,
                        "channel_windows": item_windows * len(indicator_cols),
                    }
                )

    if quality_scope == "channel":
        tasks = list(iter_quality_tasks(subset, df, indicator_cols, period, quality_max_points, quality_method))
        channel_quality = compute_quality(tasks, max_workers=max_workers)
        channel_quality, channel_thresholds = add_cells(channel_quality)
        item_quality = item_level_quality(channel_quality)
        item_quality, item_thresholds = add_cells(item_quality)
    else:
        tasks = list(iter_item_quality_tasks(subset, df, indicator_cols, period, quality_max_points, quality_method))
        item_quality = compute_quality(tasks, max_workers=max_workers)
        item_quality, item_thresholds = add_cells(item_quality)
        channel_quality = replicate_item_quality_to_channels(item_quality, indicator_cols)
        channel_thresholds = item_thresholds

    return {
        "item_lengths": pd.DataFrame(item_rows),
        "window_counts": pd.DataFrame(window_rows),
        "channel_quality": channel_quality,
        "item_quality": item_quality,
        "meta": {
            "subset": subset,
            "rows": len(df),
            "item_count": int(df["item_id"].nunique()),
            "indicator_count": len(indicator_cols),
            "period": period,
            "quality_scope": quality_scope,
            "quality_method": quality_method,
            "quality_max_points": quality_max_points,
            "channel_thresholds": channel_thresholds,
            "item_thresholds": item_thresholds,
        },
    }


def build_cell_distribution(item_quality: pd.DataFrame, channel_quality: pd.DataFrame, window_counts: pd.DataFrame) -> pd.DataFrame:
    """汇总 item 级和通道级 TSF cell 分布，并附加 train/valid/test 窗口量。"""

    rows = []
    for level, qdf in [("item", item_quality), ("channel", channel_quality)]:
        subset_labels = sorted(qdf["subset"].unique().tolist()) + ["combined"]
        for subset_label in subset_labels:
            qpart = qdf if subset_label == "combined" else qdf[qdf["subset"] == subset_label]
            wc_base = window_counts if subset_label == "combined" else window_counts[window_counts["subset"] == subset_label]
            for cell, qsub in qpart.groupby("tsf_cell"):
                if level == "item":
                    unit_count = int(qsub[["subset", "item_id"]].drop_duplicates().shape[0])
                    merge_keys = qsub[["subset", "item_id"]].drop_duplicates()
                else:
                    unit_count = int(qsub[["subset", "item_id", "channel"]].drop_duplicates().shape[0])
                    merge_keys = qsub[["subset", "item_id", "channel"]].drop_duplicates()
                for seq_len, pred_len in WINDOW_SETTINGS:
                    part = wc_base[(wc_base["seq_len"] == seq_len) & (wc_base["pred_len"] == pred_len)]
                    if level == "item":
                        matched = part.merge(merge_keys, on=["subset", "item_id"], how="inner")
                    else:
                        matched = part.merge(merge_keys, on=["subset", "item_id"], how="inner")
                    split_totals = matched.groupby("split")["item_windows"].sum().to_dict()
                    rows.append(
                        {
                            "level": level,
                            "subset": subset_label,
                            "tsf_cell": cell,
                            "unit_count": unit_count,
                            "seq_len": seq_len,
                            "pred_len": pred_len,
                            "train_windows": int(split_totals.get("train", 0)),
                            "valid_windows": int(split_totals.get("valid", 0)),
                            "test_windows": int(split_totals.get("test", 0)),
                            "supports_1shot": bool(unit_count >= 1),
                            "supports_5shot": bool(unit_count >= 5),
                            "supports_10shot": bool(unit_count >= 10),
                            "supports_50shot": bool(unit_count >= 50),
                        }
                    )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    metas: list[dict[str, object]],
    item_lengths: pd.DataFrame,
    window_counts: pd.DataFrame,
    item_quality: pd.DataFrame,
    channel_quality: pd.DataFrame,
    cell_distribution: pd.DataFrame,
) -> None:
    """写出中文审计报告。"""

    lines: list[str] = []
    lines.append("# QuitoBench 数据充分性审计报告")
    lines.append("")
    lines.append("## 1. 数据来源")
    lines.append("")
    lines.append("- 数据集：`hq-bench/quitobench`")
    lines.append("- 使用 config：`hour`、`min`")
    lines.append("- 使用 split：公开 parquet 中的 `test`，并按 README 的 `2023-07-28 00:00:00` cutoff 重建 train/test。")
    lines.append("- 明确未使用：`hq-bench/quito-corpus` 预训练 corpus。")
    lines.append("")
    lines.append("## 2. 基本规模")
    lines.append("")
    lines.append("| subset | rows | item 数 | 指标列数 | daily period |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for meta in metas:
        lines.append(
            f"| {meta['subset']} | {meta['rows']:,} | {meta['item_count']:,} | "
            f"{meta['indicator_count']} | {meta['period']} |"
        )
    lines.append("")
    lines.append("## 3. 长度分布")
    lines.append("")
    lines.append("| subset | total p50 | train p50 | valid p50 | test p50 | total min | total max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for subset, part in item_lengths.groupby("subset"):
        total = quantile_summary(part["total_length"])
        train = quantile_summary(part["train_length"])
        valid = quantile_summary(part["valid_length"])
        test = quantile_summary(part["test_length"])
        lines.append(
            f"| {subset} | {total['p50']:.0f} | {train['p50']:.0f} | {valid['p50']:.0f} | "
            f"{test['p50']:.0f} | {total['min']:.0f} | {total['max']:.0f} |"
        )
    lines.append("")
    lines.append("## 4. 窗口数量汇总")
    lines.append("")
    lines.append("| subset | seq_len | pred_len | train item windows | valid item windows | test item windows | train channel windows | valid channel windows | test channel windows |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    grouped = window_counts.groupby(["subset", "seq_len", "pred_len", "split"], as_index=False)[
        ["item_windows", "channel_windows"]
    ].sum()
    for (subset, seq_len, pred_len), part in grouped.groupby(["subset", "seq_len", "pred_len"]):
        vals = part.set_index("split")
        lines.append(
            f"| {subset} | {seq_len} | {pred_len} | "
            f"{int(vals.loc['train','item_windows']):,} | {int(vals.loc['valid','item_windows']):,} | {int(vals.loc['test','item_windows']):,} | "
            f"{int(vals.loc['train','channel_windows']):,} | {int(vals.loc['valid','channel_windows']):,} | {int(vals.loc['test','channel_windows']):,} |"
        )
    lines.append("")
    lines.append("## 5. 质量指标分布")
    lines.append("")
    quality_scope = metas[0].get("quality_scope", "item")
    if quality_scope == "item":
        lines.append("本次正式审计覆盖全部 item：每个 item 的 5 个指标列先 z-score，再沿通道取均值作为代表序列。通道级 cell 仅用于通道独立样本量口径，由 item cell 复制得到。")
    else:
        lines.append("本次正式审计采用逐通道质量指标：每个 `(item_id, ind_k)` 单独计算质量指标。")
    quality_method = metas[0].get("quality_method", "light")
    if quality_method == "light":
        lines.append("质量指标使用轻量近似：forecastability=FFT 频谱熵，seasonality=日周期滞后自相关，trend=线性趋势 R2。该口径用于数据充分性审计，不等同于 Quito/STL 精确质量标签。")
    else:
        lines.append("质量指标使用 Quito 原生 `evaluate_series` / STL 口径。")
    lines.append(f"质量指标计算使用等距降采样上限：`{metas[0].get('quality_max_points')}` 点；长度和窗口数仍使用完整原始序列。")
    lines.append("")
    lines.append("### Item 级")
    lines.append("")
    lines.append("| subset | metric | min | p25 | p50 | p75 | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for subset, part in item_quality.groupby("subset"):
        for metric in ["forecastability", "seasonality_strength", "trend_strength"]:
            s = quantile_summary(part[metric])
            lines.append(f"| {subset} | {metric} | {s['min']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | {s['p75']:.4f} | {s['max']:.4f} |")
    lines.append("")
    lines.append("### 通道级")
    lines.append("")
    lines.append("| subset | metric | min | p25 | p50 | p75 | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for subset, part in channel_quality.groupby("subset"):
        for metric in ["forecastability", "seasonality_strength", "trend_strength"]:
            s = quantile_summary(part[metric])
            lines.append(f"| {subset} | {metric} | {s['min']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | {s['p75']:.4f} | {s['max']:.4f} |")
    lines.append("")
    lines.append("## 6. TSF cell 分布")
    lines.append("")
    lines.append("TSF cell 使用本次审计中位数阈值重新构造。公开 README 另说明 1,290 条 item 序列已按 8 个 TSF regime cell 分层均衡，约 160 条序列/cell。")
    lines.append("")
    lines.append("| level | subset | seq_len | cell 数 | 最小 unit 数 | 最大 unit 数 | 最小 train windows | 最小 valid windows | 最小 test windows | 50-shot 全支持 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for (level, subset, seq_len), part in cell_distribution.groupby(["level", "subset", "seq_len"]):
        lines.append(
            f"| {level} | {subset} | {seq_len} | {part['tsf_cell'].nunique()} | "
            f"{int(part['unit_count'].min())} | {int(part['unit_count'].max())} | "
            f"{int(part['train_windows'].min())} | {int(part['valid_windows'].min())} | {int(part['test_windows'].min())} | "
            f"{bool(part['supports_50shot'].all())} |"
        )
    lines.append("")
    lines.append("## 7. 充分性结论")
    lines.append("")
    lines.append("- 路线 1：QuitoBench 的普通样本级融合/路由验证在窗口数量上足够；即使使用 `seq_len=336,pred_len=96`，整体 item 窗口仍有数万到数百万规模，通道独立口径进一步放大 5 倍。")
    combined_item = cell_distribution[(cell_distribution["level"] == "item") & (cell_distribution["subset"] == "combined")]
    combined_channel = cell_distribution[(cell_distribution["level"] == "channel") & (cell_distribution["subset"] == "combined")]
    lines.append("- 路线 2：公开 README 声明 1,290 条 item 序列在 8 个 TSF regime cell 上分层均衡，约 160 条序列/cell，这是支持 QuitoBench-only 路线 2 的主要证据。")
    lines.append(
        f"- 轻量 proxy cell 作为复核：combined item 口径覆盖 {combined_item['tsf_cell'].nunique()} 个 cell，"
        f"最小 cell 为 {int(combined_item['unit_count'].min())} 个 item；combined 通道口径最小 cell 为 "
        f"{int(combined_channel['unit_count'].min())} 个通道样本。subset 内 proxy cell 可能不均衡，因此路线 2 主报告应优先使用官方 TSF regime 或合并/重分位策略。"
    )
    lines.append("- 当前仍不能回答专家之间是否存在 oracle gap，因为这需要先生成多个专家在同一窗口上的预测缓存；该项应作为后续专家 profiling 的第一步，而不是数据充分性本身的阻塞项。")
    lines.append("")
    lines.append("## 8. 建议")
    lines.append("")
    lines.append("- 第一阶段可以只使用 QuitoBench 推进路线 1 和路线 2。")
    lines.append("- 路线 1 使用普通 train/valid/test 窗口切分，不使用 TSF 标签训练 router。")
    lines.append("- 路线 2 使用 item 级 TSF cell 作为主报告口径，通道级 cell 作为通道独立伪图像策略的补充审计口径；后续需优先定位官方 TSF regime 标签，若不可得则使用全局分位/合并稀疏 cell 的 proxy cell。")
    lines.append("- TimeFuse 传统数据集暂不进入第一阶段，仅保留为外部泛化检查。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 QuitoBench benchmark 数据充分性。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/hf/hq-bench/quitobench/v20260315")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--quality-scope",
        choices=["item", "channel"],
        default="item",
        help="item 为默认正式审计口径；channel 会逐通道计算 STL，耗时较长。",
    )
    parser.add_argument(
        "--quality-method",
        choices=["light", "stl"],
        default="light",
        help="light 为快速近似质量指标；stl 为 Quito 原生 STL 指标，耗时较长。",
    )
    parser.add_argument(
        "--quality-max-points",
        type=int,
        default=2048,
        help="质量指标计算的等距降采样点数上限；<=0 表示使用全长序列。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for subset in ["hour", "min"]:
        path = args.data_dir / f"test_{subset}-00001-of-00001.parquet"
        results.append(
            audit_subset(
                subset,
                path,
                max_workers=args.max_workers,
                quality_scope=args.quality_scope,
                quality_max_points=args.quality_max_points,
                quality_method=args.quality_method,
            )
        )

    item_lengths = pd.concat([r["item_lengths"] for r in results], ignore_index=True)
    window_counts = pd.concat([r["window_counts"] for r in results], ignore_index=True)
    channel_quality = pd.concat([r["channel_quality"] for r in results], ignore_index=True)
    item_quality = pd.concat([r["item_quality"] for r in results], ignore_index=True)
    metas = [r["meta"] for r in results]

    cell_cols = ["trend_bin", "seasonality_bin", "forecastability_bin", "tsf_cell"]
    item_quality = item_quality.drop(columns=[c for c in cell_cols if c in item_quality.columns])
    channel_quality = channel_quality.drop(columns=[c for c in cell_cols if c in channel_quality.columns])
    item_quality, _ = add_cells(item_quality)
    channel_quality, _ = add_cells(channel_quality)

    cell_distribution = build_cell_distribution(item_quality, channel_quality, window_counts)

    item_lengths.to_csv(args.output_dir / "quitobench_item_lengths.csv", index=False)
    window_counts.to_csv(args.output_dir / "quitobench_window_counts.csv", index=False)
    channel_quality.to_csv(args.output_dir / "quitobench_channel_quality.csv", index=False)
    item_quality.to_csv(args.output_dir / "quitobench_item_quality.csv", index=False)
    cell_distribution.to_csv(args.output_dir / "quitobench_cell_distribution.csv", index=False)

    write_report(
        args.output_dir / "quitobench_sufficiency_report.md",
        metas=metas,
        item_lengths=item_lengths,
        window_counts=window_counts,
        item_quality=item_quality,
        channel_quality=channel_quality,
        cell_distribution=cell_distribution,
    )

    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
