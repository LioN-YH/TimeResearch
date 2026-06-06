"""Stage 0.1：QuitoBench item 级全长 Quito/STL 精确质量审计。

本脚本只读取 `hq-bench/quitobench` benchmark parquet，不读取 Quito 预训练
corpus。默认口径为 item 级：每个 item 的 L x C 指标矩阵先逐通道 z-score，
再沿通道取均值，得到一条代表序列后调用 Quito 原生
`quito.utils.dataset_quality.evaluate_series`。

设计目标：
- 全长 STL，不做降采样。
- 分批进度输出。
- 分批写中间 CSV，支持断点续跑。
- 只生成质量指标和 TSF cell 审计，不实现 router、不做伪图像。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))


CUTOFF = pd.Timestamp("2023-07-28 00:00:00")
WINDOW_SETTINGS = [(96, 96), (192, 96), (336, 96)]
QUALITY_COLUMNS = [
    "subset",
    "item_id",
    "channel",
    "quality_method",
    "quality_points",
    "effective_length",
    "missing_ratio",
    "forecastability",
    "seasonality_strength",
    "trend_strength",
    "cv",
    "adf_stat",
    "hurst",
    "seconds",
    "period",
    "compute_adf",
    "compute_hurst",
]


@dataclass(frozen=True)
class ItemQualityTask:
    """单个 item 的全长 STL 质量任务。

    输入 shape：
    - values: `L`，由原始 `L x C` 指标矩阵 z-score 后按通道均值得到。

    输出：
    - Quito `evaluate_series` 的 item 级质量指标。

    CPU/GPU 说明：
    - 全流程 CPU 执行，不发生 GPU 数据迁移。
    """

    subset: str
    item_id: int
    values: np.ndarray
    period: int
    compute_adf: bool
    compute_hurst: bool


def representative_series(item_df: pd.DataFrame, indicator_cols: list[str]) -> np.ndarray:
    """把单个 item 的多通道矩阵转成 item 级代表序列。

    输入 shape：
    - `item_df[indicator_cols]`: `L x C`，当前 QuitoBench 中 C=5。

    输出 shape：
    - `representative`: `L`。

    说明：
    - 先对每个通道独立 z-score，再沿通道求均值，避免量纲支配 item 级质量。
    - 这里只用于 TSF cell 审计，不改变后续默认通道独立伪图像策略。
    """

    ordered = item_df.sort_values("date_time")
    values = ordered[indicator_cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True) + 1e-8
    return np.nanmean((values - mean) / std, axis=1)


def split_lengths_for_item(item_df: pd.DataFrame) -> tuple[int, int, int]:
    """按 Stage 0 已采用的 QuitoBench cutoff 重建 train/valid/test 长度。"""

    dates = pd.to_datetime(item_df["date_time"])
    pre_cutoff_len = int((dates < CUTOFF).sum())
    test_len = int((dates >= CUTOFF).sum())
    valid_len = int(pre_cutoff_len * 0.2)
    train_len = pre_cutoff_len - valid_len
    return train_len, valid_len, test_len


def window_count(length: int, seq_len: int, pred_len: int) -> int:
    """计算一条 item 序列在给定窗口设置下的滑动窗口数。"""

    return max(int(length) - int(seq_len) - int(pred_len) + 1, 0)


def iter_tasks_and_windows(data_dir: Path) -> tuple[list[ItemQualityTask], pd.DataFrame, dict[str, object]]:
    """读取 QuitoBench hour/min parquet，生成 item 级任务和窗口统计。"""

    tasks: list[ItemQualityTask] = []
    window_rows: list[dict[str, object]] = []
    meta: dict[str, object] = {
        "dataset": "hq-bench/quitobench",
        "subsets": [],
        "rows": {},
        "items": {},
        "indicator_count": {},
    }

    for subset, period in [("hour", 24), ("min", 144)]:
        path = data_dir / f"test_{subset}-00001-of-00001.parquet"
        df = pd.read_parquet(path)
        df["date_time"] = pd.to_datetime(df["date_time"])
        indicator_cols = [c for c in df.columns if c.startswith("ind_")]
        meta["subsets"].append(subset)
        meta["rows"][subset] = int(len(df))
        meta["items"][subset] = int(df["item_id"].nunique())
        meta["indicator_count"][subset] = int(len(indicator_cols))

        for item_id, item_df in df.groupby("item_id", sort=True):
            ordered = item_df.sort_values("date_time")
            tasks.append(
                ItemQualityTask(
                    subset=subset,
                    item_id=int(item_id),
                    values=representative_series(ordered, indicator_cols),
                    period=period,
                    compute_adf=False,
                    compute_hurst=True,
                )
            )
            train_len, valid_len, test_len = split_lengths_for_item(ordered)
            for seq_len, pred_len in WINDOW_SETTINGS:
                for split_name, split_len in [("train", train_len), ("valid", valid_len), ("test", test_len)]:
                    window_rows.append(
                        {
                            "subset": subset,
                            "item_id": int(item_id),
                            "seq_len": seq_len,
                            "pred_len": pred_len,
                            "split": split_name,
                            "split_length": split_len,
                            "item_windows": window_count(split_len, seq_len, pred_len),
                        }
                    )

    return tasks, pd.DataFrame(window_rows), meta


def evaluate_item_task(task: ItemQualityTask) -> dict[str, object]:
    """调用 Quito 原生 `evaluate_series` 计算单个 item 的全长质量指标。"""

    from quito.utils.dataset_quality import evaluate_series

    period = min(task.period, max(2, len(task.values) // 2))
    t0 = time.perf_counter()
    result = evaluate_series(
        task.values,
        period=period,
        compute_adf=task.compute_adf,
        compute_hurst=task.compute_hurst,
    )
    seconds = time.perf_counter() - t0
    return {
        "subset": task.subset,
        "item_id": task.item_id,
        "channel": "item_mean_z",
        "quality_method": "quito_evaluate_series_full_stl",
        "quality_points": int(len(task.values)),
        "effective_length": int(result.eff_length),
        "missing_ratio": float(result.missing_ratio),
        "forecastability": float(result.forecastability),
        "seasonality_strength": float(result.season_strength),
        "trend_strength": float(result.trend_strength),
        "cv": float(result.cv),
        "adf_stat": float(result.adf_stat) if result.adf_stat is not None else math.nan,
        "hurst": float(result.hurst) if result.hurst is not None else math.nan,
        "seconds": float(seconds),
        "period": int(period),
        "compute_adf": bool(task.compute_adf),
        "compute_hurst": bool(task.compute_hurst),
    }


def filter_completed_tasks(
    tasks: Iterable[Mapping[str, object] | ItemQualityTask],
    existing_quality: pd.DataFrame | None,
) -> list[Mapping[str, object] | ItemQualityTask]:
    """根据已有 CSV 跳过已完成的 `(subset, item_id)`。

    只用 subset 和 item_id 判断完成状态，避免同一 item 重复写入。
    """

    if existing_quality is None or existing_quality.empty:
        return list(tasks)
    completed = {
        (str(row.subset), int(row.item_id))
        for row in existing_quality[["subset", "item_id"]].dropna().itertuples(index=False)
    }
    remaining: list[Mapping[str, object] | ItemQualityTask] = []
    for task in tasks:
        subset = getattr(task, "subset", None) if not isinstance(task, Mapping) else task["subset"]
        item_id = getattr(task, "item_id", None) if not isinstance(task, Mapping) else task["item_id"]
        if (str(subset), int(item_id)) not in completed:
            remaining.append(task)
    return remaining


def quantile_summary(values: pd.Series) -> dict[str, float]:
    """生成报告用分位数摘要。"""

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


def add_cells(quality_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """使用全局中位数阈值构造 `trend x seasonality x forecastability` 8-cell。"""

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
    df["forecastability_bin"] = np.where(
        df["forecastability"] >= thresholds["forecastability_threshold"], "highF", "lowF"
    )
    df["tsf_cell"] = df["trend_bin"] + "_" + df["seasonality_bin"] + "_" + df["forecastability_bin"]
    return df, thresholds


def build_cell_distribution(
    item_quality: pd.DataFrame,
    window_counts: pd.DataFrame,
    window_settings: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """汇总 item 级 STL cell 分布，并附加 train/valid/test 窗口量。"""

    settings = window_settings or WINDOW_SETTINGS
    rows: list[dict[str, object]] = []
    subset_labels = sorted(item_quality["subset"].unique().tolist()) + ["combined"]
    for subset_label in subset_labels:
        qpart = item_quality if subset_label == "combined" else item_quality[item_quality["subset"] == subset_label]
        wc_base = window_counts if subset_label == "combined" else window_counts[window_counts["subset"] == subset_label]
        for cell, qsub in qpart.groupby("tsf_cell"):
            merge_keys = qsub[["subset", "item_id"]].drop_duplicates()
            unit_count = int(merge_keys.shape[0])
            for seq_len, pred_len in settings:
                part = wc_base[(wc_base["seq_len"] == seq_len) & (wc_base["pred_len"] == pred_len)]
                matched = part.merge(merge_keys, on=["subset", "item_id"], how="inner")
                split_totals = matched.groupby("split")["item_windows"].sum().to_dict()
                rows.append(
                    {
                        "level": "item",
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


def read_existing_quality(path: Path) -> pd.DataFrame:
    """读取已有中间 CSV；不存在时返回空表。"""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=QUALITY_COLUMNS)
    return pd.read_csv(path)


def write_quality_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """按固定列顺序写 item 级质量 CSV。"""

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=QUALITY_COLUMNS)
    for col in QUALITY_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan
    df = df[QUALITY_COLUMNS].sort_values(["subset", "item_id"]).drop_duplicates(["subset", "item_id"], keep="last")
    df.to_csv(path, index=False)


def compute_quality_with_progress(
    tasks: list[ItemQualityTask],
    output_path: Path,
    max_workers: int,
    batch_size: int,
    resume: bool,
) -> pd.DataFrame:
    """计算全量 item 级 STL 质量，并按批次写中间 CSV。"""

    existing = read_existing_quality(output_path) if resume else pd.DataFrame(columns=QUALITY_COLUMNS)
    completed_rows = existing.to_dict("records")
    remaining = filter_completed_tasks(tasks, existing)
    total = len(tasks)
    done_before = total - len(remaining)
    print(
        f"[start] total={total}, completed={done_before}, remaining={len(remaining)}, "
        f"workers={max_workers}, batch_size={batch_size}",
        flush=True,
    )
    if not remaining:
        quality = existing.copy()
        quality, _ = add_cells(quality)
        return quality

    rows = completed_rows
    completed_now = 0
    t0 = time.perf_counter()
    last_write = time.perf_counter()

    if max_workers <= 1:
        iterator = ((task, evaluate_item_task(task)) for task in remaining)  # type: ignore[arg-type]
        for _, row in iterator:
            rows.append(row)
            completed_now += 1
            if completed_now % batch_size == 0 or done_before + completed_now == total:
                write_quality_csv(output_path, rows)
                elapsed = time.perf_counter() - t0
                avg = elapsed / max(completed_now, 1)
                print(
                    f"[progress] done={done_before + completed_now}/{total}, "
                    f"batch_seconds={time.perf_counter() - last_write:.1f}, "
                    f"avg_new_seconds={avg:.2f}, csv={output_path}",
                    flush=True,
                )
                last_write = time.perf_counter()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(evaluate_item_task, task) for task in remaining]  # type: ignore[arg-type]
            for future in as_completed(futures):
                rows.append(future.result())
                completed_now += 1
                if completed_now % batch_size == 0 or done_before + completed_now == total:
                    write_quality_csv(output_path, rows)
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / max(completed_now, 1)
                    print(
                        f"[progress] done={done_before + completed_now}/{total}, "
                        f"batch_seconds={time.perf_counter() - last_write:.1f}, "
                        f"avg_new_seconds={avg:.2f}, csv={output_path}",
                        flush=True,
                    )
                    last_write = time.perf_counter()

    quality = pd.read_csv(output_path)
    quality, _ = add_cells(quality)
    return quality


def compare_with_proxy(stl_quality: pd.DataFrame, proxy_path: Path) -> dict[str, object]:
    """和 Stage 0 light proxy 结果对比。"""

    if not proxy_path.exists():
        return {"available": False}
    proxy = pd.read_csv(proxy_path)
    merge_cols = ["subset", "item_id"]
    keep_cols = merge_cols + ["forecastability", "seasonality_strength", "trend_strength", "tsf_cell"]
    merged = stl_quality[keep_cols].merge(
        proxy[keep_cols],
        on=merge_cols,
        how="inner",
        suffixes=("_stl", "_proxy"),
    )
    if merged.empty:
        return {"available": False}
    correlations = {}
    for metric in ["forecastability", "seasonality_strength", "trend_strength"]:
        correlations[metric] = float(merged[f"{metric}_stl"].corr(merged[f"{metric}_proxy"], method="spearman"))
    return {
        "available": True,
        "matched_items": int(len(merged)),
        "same_cell_count": int((merged["tsf_cell_stl"] == merged["tsf_cell_proxy"]).sum()),
        "same_cell_ratio": float((merged["tsf_cell_stl"] == merged["tsf_cell_proxy"]).mean()),
        "spearman": correlations,
    }


def write_report(
    output_path: Path,
    item_quality: pd.DataFrame,
    cell_distribution: pd.DataFrame,
    thresholds: dict[str, float],
    proxy_compare: dict[str, object],
    meta: dict[str, object],
    command: str,
) -> None:
    """写出 Stage 0.1 中文报告。"""

    lines: list[str] = []
    lines.append("# QuitoBench item 级全长 STL 精确质量审计报告")
    lines.append("")
    lines.append("## 1. 数据来源与口径")
    lines.append("")
    lines.append("- 数据集：`hq-bench/quitobench` benchmark。")
    lines.append("- 使用 config：`hour`、`min`。")
    lines.append("- 明确未使用：`hq-bench/quito-corpus` 预训练 corpus。")
    lines.append("- 质量函数：`quito.utils.dataset_quality.evaluate_series`。")
    lines.append("- 质量口径：item 级全长 STL；每个 item 的指标列先 z-score，再沿通道取均值。")
    lines.append("- 当前不做通道级全长 STL、不做伪图像、不做 router。")
    lines.append("")
    lines.append("## 2. 执行命令")
    lines.append("")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## 3. 基本规模")
    lines.append("")
    lines.append("| subset | rows | item 数 | 指标列数 | period |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for subset in ["hour", "min"]:
        period = 24 if subset == "hour" else 144
        lines.append(
            f"| {subset} | {meta['rows'][subset]:,} | {meta['items'][subset]:,} | "
            f"{meta['indicator_count'][subset]} | {period} |"
        )
    lines.append("")
    lines.append("## 4. STL 质量指标分布")
    lines.append("")
    lines.append("| subset | metric | min | p25 | p50 | p75 | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for subset, part in item_quality.groupby("subset"):
        for metric in ["forecastability", "seasonality_strength", "trend_strength", "hurst", "seconds"]:
            s = quantile_summary(part[metric])
            lines.append(
                f"| {subset} | {metric} | {s['min']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | "
                f"{s['p75']:.4f} | {s['max']:.4f} |"
            )
    lines.append("")
    lines.append("## 5. TSF cell 构造阈值")
    lines.append("")
    lines.append("| threshold | value |")
    lines.append("| --- | ---: |")
    for key, value in thresholds.items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.append("")
    lines.append("## 6. STL TSF cell 分布")
    lines.append("")
    lines.append("| subset | seq_len | cell 数 | 最小 item 数 | 最大 item 数 | 最小 train windows | 最小 valid windows | 最小 test windows | 50-shot 全支持 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for (subset, seq_len), part in cell_distribution.groupby(["subset", "seq_len"]):
        lines.append(
            f"| {subset} | {seq_len} | {part['tsf_cell'].nunique()} | "
            f"{int(part['unit_count'].min())} | {int(part['unit_count'].max())} | "
            f"{int(part['train_windows'].min())} | {int(part['valid_windows'].min())} | "
            f"{int(part['test_windows'].min())} | {bool(part['supports_50shot'].all())} |"
        )
    lines.append("")
    lines.append("## 7. 与 Stage 0 light proxy 对比")
    lines.append("")
    if proxy_compare.get("available"):
        spearman = proxy_compare["spearman"]
        lines.append(f"- 匹配 item 数：{proxy_compare['matched_items']:,}。")
        lines.append(
            f"- STL cell 与 proxy cell 完全一致：{proxy_compare['same_cell_count']:,} "
            f"({proxy_compare['same_cell_ratio']:.2%})。"
        )
        lines.append(f"- Spearman 相关：forecastability={spearman['forecastability']:.4f}，seasonality={spearman['seasonality_strength']:.4f}，trend={spearman['trend_strength']:.4f}。")
    else:
        lines.append("- 未找到 Stage 0 proxy CSV，未执行对比。")
    lines.append("")
    lines.append("## 8. 结论")
    lines.append("")
    combined = cell_distribution[cell_distribution["subset"] == "combined"]
    lines.append(
        f"- item 级全长 STL 精确质量指标已覆盖 {len(item_quality):,} 个 QuitoBench item，"
        f"combined 口径覆盖 {combined['tsf_cell'].nunique()} 个 TSF cell。"
    )
    lines.append("- STL 指标可作为 Stage 0.5 固化最终 cell 构造规则的主要 proxy 候选；官方 TSF regime 标签仍需优先定位。")
    lines.append("- 通道级全长 STL 预计耗时显著更长，应作为单独长实验处理。")
    lines.append("")
    lines.append("## 9. 下一步计划")
    lines.append("")
    lines.append("1. 进入 Stage 0.5，定位 QuitoBench 官方 TSF regime/cell 标签是否存在。")
    lines.append("2. 如果官方标签不可得，基于本报告 STL 指标和 Stage 0 proxy 对比固化最终 cell 构造规则。")
    lines.append("3. 当前仍不实现 router。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0.1 QuitoBench item 级全长 STL 精确质量审计。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/hf/hq-bench/quitobench/v20260315")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true", help="忽略已有中间 CSV，从头重算。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = args.output_dir / "quitobench_item_quality_stl.csv"
    cell_path = args.output_dir / "quitobench_stl_cell_distribution.csv"
    report_path = args.output_dir / "quitobench_stl_quality_report.md"
    proxy_path = args.output_dir / "quitobench_item_quality.csv"

    tasks, window_counts, meta = iter_tasks_and_windows(args.data_dir)
    item_quality = compute_quality_with_progress(
        tasks=tasks,
        output_path=quality_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    item_quality, thresholds = add_cells(item_quality)
    item_quality.to_csv(quality_path, index=False)
    cell_distribution = build_cell_distribution(item_quality, window_counts)
    cell_distribution.to_csv(cell_path, index=False)

    command = (
        "conda run -n quito python tools/quitobench_item_stl_quality_audit.py "
        f"--max-workers {args.max_workers} --batch-size {args.batch_size}"
    )
    proxy_compare = compare_with_proxy(item_quality, proxy_path)
    write_report(
        report_path,
        item_quality=item_quality,
        cell_distribution=cell_distribution,
        thresholds=thresholds,
        proxy_compare=proxy_compare,
        meta=meta,
        command=command,
    )
    print(f"[done] wrote {quality_path}", flush=True)
    print(f"[done] wrote {cell_path}", flush=True)
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
