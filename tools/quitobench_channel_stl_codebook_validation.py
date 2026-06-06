"""Stage 0.7：QuitoBench 通道级全长 STL 官方 codebook 验证。

本脚本只读取 `hq-bench/quitobench` benchmark，按 `(subset, item_id, ind_k)`
运行 Quito 原生 `evaluate_series`，保留通道级中间结果，再按论文 multivariate
TSF 口径对 5 个 channel 的 T/S/F 指标求均值并用 `tau=0.4` 二值化。

本阶段只验证 Stage 0.6b codebook，不重新定义官方标签，不实现 router。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))


TSF_CELLS = [
    "highT_highS_highF",
    "highT_highS_lowF",
    "highT_lowS_highF",
    "highT_lowS_lowF",
    "lowT_highS_highF",
    "lowT_highS_lowF",
    "lowT_lowS_highF",
    "lowT_lowS_lowF",
]
CHANNEL_QUALITY_COLUMNS = [
    "subset",
    "item_id",
    "official_cluster_code",
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
ITEM_MEAN_COLUMNS = [
    "subset",
    "item_id",
    "official_cluster_code",
    "channel_count",
    "complete_channel_count",
    "trend_strength_channel_mean",
    "seasonality_strength_channel_mean",
    "forecastability_channel_mean",
    "paper_like_tsf_cell",
]


@dataclass(frozen=True)
class ChannelQualityTask:
    """单个 QuitoBench 通道的 full-length STL 质量任务。"""

    subset: str
    item_id: int
    official_cluster_code: int
    channel: str
    values: np.ndarray
    period: int
    compute_adf: bool
    compute_hurst: bool


def cell_from_thresholds(
    trend_strength: float,
    seasonality_strength: float,
    forecastability: float,
    tau: float = 0.4,
) -> str:
    """按论文固定阈值构造 TSF cell：指标 `> tau` 为 high。"""

    t = "highT" if float(trend_strength) > tau else "lowT"
    s = "highS" if float(seasonality_strength) > tau else "lowS"
    f = "highF" if float(forecastability) > tau else "lowF"
    return f"{t}_{s}_{f}"


def _split_cell(cell: str) -> tuple[str, str, str]:
    parts = str(cell).split("_")
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def iter_channel_tasks(
    data_dir: Path,
    max_items_per_subset: int | None = None,
) -> tuple[list[ChannelQualityTask], dict[str, object]]:
    """读取 QuitoBench hour/min parquet，生成通道级 full-length 任务。"""

    tasks: list[ChannelQualityTask] = []
    meta: dict[str, object] = {
        "dataset": "hq-bench/quitobench",
        "subsets": [],
        "rows": {},
        "items": {},
        "indicator_count": {},
    }
    for subset, period in [("hour", 24), ("min", 144)]:
        path = data_dir / f"test_{subset}-00001-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少 QuitoBench benchmark parquet：{path}")
        df = pd.read_parquet(path)
        df["date_time"] = pd.to_datetime(df["date_time"])
        indicator_cols = [c for c in df.columns if c.startswith("ind_")]
        if "cluster" not in df.columns:
            raise ValueError(f"{path} 缺少官方 cluster 列，请使用 revision 17362dcb 的 benchmark 文件。")
        meta["subsets"].append(subset)
        meta["rows"][subset] = int(len(df))
        meta["items"][subset] = int(df["item_id"].nunique())
        meta["indicator_count"][subset] = int(len(indicator_cols))

        grouped = df.groupby("item_id", sort=True)
        if max_items_per_subset is not None:
            grouped = list(grouped)[: int(max_items_per_subset)]
        for item_id, item_df in grouped:
            ordered = item_df.sort_values("date_time")
            cluster_values = ordered["cluster"].dropna().unique()
            if len(cluster_values) != 1:
                raise ValueError(f"{subset}/{item_id} 的 cluster 不唯一：{cluster_values}")
            cluster_code = int(cluster_values[0])
            for channel in indicator_cols:
                values = ordered[channel].to_numpy(dtype=float)
                tasks.append(
                    ChannelQualityTask(
                        subset=subset,
                        item_id=int(item_id),
                        official_cluster_code=cluster_code,
                        channel=channel,
                        values=values,
                        period=period,
                        compute_adf=False,
                        compute_hurst=True,
                    )
                )
    return tasks, meta


def evaluate_channel_task(task: ChannelQualityTask) -> dict[str, object]:
    """调用 Quito 原生 `evaluate_series` 计算单个通道的质量指标。"""

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
        "official_cluster_code": task.official_cluster_code,
        "channel": task.channel,
        "quality_method": "quito_evaluate_series_full_stl_channel",
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


def read_existing_channel_quality(path: Path) -> pd.DataFrame:
    """读取通道级中间 CSV；不存在时返回空表。"""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=CHANNEL_QUALITY_COLUMNS)
    return pd.read_csv(path)


def filter_completed_channel_tasks(
    tasks: Iterable[Mapping[str, object] | ChannelQualityTask],
    existing_quality: pd.DataFrame | None,
) -> list[Mapping[str, object] | ChannelQualityTask]:
    """按 `(subset, item_id, channel)` 跳过已完成通道任务。"""

    if existing_quality is None or existing_quality.empty:
        return list(tasks)
    completed = {
        (str(row.subset), int(row.item_id), str(row.channel))
        for row in existing_quality[["subset", "item_id", "channel"]].dropna().itertuples(index=False)
    }
    remaining: list[Mapping[str, object] | ChannelQualityTask] = []
    for task in tasks:
        subset = getattr(task, "subset", None) if not isinstance(task, Mapping) else task["subset"]
        item_id = getattr(task, "item_id", None) if not isinstance(task, Mapping) else task["item_id"]
        channel = getattr(task, "channel", None) if not isinstance(task, Mapping) else task["channel"]
        if (str(subset), int(item_id), str(channel)) not in completed:
            remaining.append(task)
    return remaining


def write_channel_quality_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """按固定列顺序写通道级质量 CSV，并用 resume key 去重。"""

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=CHANNEL_QUALITY_COLUMNS)
    for col in CHANNEL_QUALITY_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan
    df = df[CHANNEL_QUALITY_COLUMNS].sort_values(["subset", "item_id", "channel"])
    df = df.drop_duplicates(["subset", "item_id", "channel"], keep="last")
    df.to_csv(path, index=False)


def compute_channel_quality_with_progress(
    tasks: list[ChannelQualityTask],
    output_path: Path,
    max_workers: int,
    batch_size: int,
    resume: bool,
) -> pd.DataFrame:
    """计算通道级 STL 质量，并分批写中间 CSV。"""

    existing = read_existing_channel_quality(output_path) if resume else pd.DataFrame(columns=CHANNEL_QUALITY_COLUMNS)
    rows = existing.to_dict("records")
    remaining = filter_completed_channel_tasks(tasks, existing)
    total = len(tasks)
    done_before = total - len(remaining)
    print(
        f"[start] total={total}, completed={done_before}, remaining={len(remaining)}, "
        f"workers={max_workers}, batch_size={batch_size}",
        flush=True,
    )
    if not remaining:
        return existing.copy()

    completed_now = 0
    t0 = time.perf_counter()
    last_write = time.perf_counter()

    def maybe_write() -> None:
        nonlocal last_write
        write_channel_quality_csv(output_path, rows)
        elapsed = time.perf_counter() - t0
        avg = elapsed / max(completed_now, 1)
        print(
            f"[progress] done={done_before + completed_now}/{total}, "
            f"batch_seconds={time.perf_counter() - last_write:.1f}, "
            f"avg_new_seconds={avg:.2f}, csv={output_path}",
            flush=True,
        )
        last_write = time.perf_counter()

    if max_workers <= 1:
        for task in remaining:
            rows.append(evaluate_channel_task(task))  # type: ignore[arg-type]
            completed_now += 1
            if completed_now % batch_size == 0 or done_before + completed_now == total:
                maybe_write()
    else:
        remaining_iter = iter(remaining)
        pending = set()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for _ in range(max_workers * 2):
                try:
                    pending.add(pool.submit(evaluate_channel_task, next(remaining_iter)))  # type: ignore[arg-type]
                except StopIteration:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.append(future.result())
                    completed_now += 1
                    try:
                        pending.add(pool.submit(evaluate_channel_task, next(remaining_iter)))  # type: ignore[arg-type]
                    except StopIteration:
                        pass
                    if completed_now % batch_size == 0 or done_before + completed_now == total:
                        maybe_write()

    return pd.read_csv(output_path)


def build_item_channel_mean(channel_quality: pd.DataFrame, tau: float = 0.4) -> pd.DataFrame:
    """对每个 item 的通道级 T/S/F 指标求均值并生成 paper-like cell。"""

    rows: list[dict[str, object]] = []
    for (subset, item_id), part in channel_quality.groupby(["subset", "item_id"], sort=True):
        if "official_cluster_code" in part.columns:
            cluster_values = pd.to_numeric(part["official_cluster_code"], errors="coerce").dropna().unique()
            cluster_code = int(cluster_values[0]) if len(cluster_values) else -1
        else:
            cluster_code = -1
        metric_part = part[["trend_strength", "seasonality_strength", "forecastability"]].apply(pd.to_numeric, errors="coerce")
        complete_channel_count = int(metric_part.notna().all(axis=1).sum())
        trend_mean = float(metric_part["trend_strength"].mean())
        seasonality_mean = float(metric_part["seasonality_strength"].mean())
        forecastability_mean = float(metric_part["forecastability"].mean())
        rows.append(
            {
                "subset": subset,
                "item_id": int(item_id),
                "official_cluster_code": cluster_code,
                "channel_count": int(part["channel"].nunique()),
                "complete_channel_count": complete_channel_count,
                "trend_strength_channel_mean": trend_mean,
                "seasonality_strength_channel_mean": seasonality_mean,
                "forecastability_channel_mean": forecastability_mean,
                "paper_like_tsf_cell": cell_from_thresholds(trend_mean, seasonality_mean, forecastability_mean, tau=tau),
            }
        )
    return pd.DataFrame(rows, columns=ITEM_MEAN_COLUMNS)


def compare_with_official_codebook(item_mean: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """把 paper-like cell 与 Stage 0.6b 官方 codebook cell 比较。"""

    merged = item_mean.merge(
        codebook[["official_cluster_code", "official_tsf_cell"]],
        on="official_cluster_code",
        how="left",
        validate="many_to_one",
    )
    official_parts = merged["official_tsf_cell"].apply(_split_cell)
    paper_parts = merged["paper_like_tsf_cell"].apply(_split_cell)
    merged["official_trend_bin"] = [p[0] for p in official_parts]
    merged["official_seasonality_bin"] = [p[1] for p in official_parts]
    merged["official_forecastability_bin"] = [p[2] for p in official_parts]
    merged["paper_like_trend_bin"] = [p[0] for p in paper_parts]
    merged["paper_like_seasonality_bin"] = [p[1] for p in paper_parts]
    merged["paper_like_forecastability_bin"] = [p[2] for p in paper_parts]
    merged["item_exact_match"] = merged["paper_like_tsf_cell"] == merged["official_tsf_cell"]
    merged["trend_match"] = merged["paper_like_trend_bin"] == merged["official_trend_bin"]
    merged["seasonality_match"] = merged["paper_like_seasonality_bin"] == merged["official_seasonality_bin"]
    merged["forecastability_match"] = merged["paper_like_forecastability_bin"] == merged["official_forecastability_bin"]
    return merged


def build_cluster_summary(validation: pd.DataFrame) -> pd.DataFrame:
    """按官方 cluster 汇总 exact/dim match 与 paper-like 众数。"""

    rows: list[dict[str, object]] = []
    for code, part in validation.groupby("official_cluster_code", sort=True):
        mode = part["paper_like_tsf_cell"].mode()
        rows.append(
            {
                "official_cluster_code": int(code),
                "official_tsf_cell": part["official_tsf_cell"].iloc[0],
                "item_count": int(len(part)),
                "paper_like_mode_cell": mode.iloc[0] if not mode.empty else "",
                "paper_like_mode_ratio": float((part["paper_like_tsf_cell"] == mode.iloc[0]).mean()) if not mode.empty else math.nan,
                "item_exact_match_ratio": float(part["item_exact_match"].mean()),
                "trend_match_ratio": float(part["trend_match"].mean()),
                "seasonality_match_ratio": float(part["seasonality_match"].mean()),
                "forecastability_match_ratio": float(part["forecastability_match"].mean()),
                "trend_strength_channel_mean": float(part["trend_strength_channel_mean"].mean()),
                "seasonality_strength_channel_mean": float(part["seasonality_strength_channel_mean"].mean()),
                "forecastability_channel_mean": float(part["forecastability_channel_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_confusion_matrix(validation: pd.DataFrame) -> pd.DataFrame:
    """生成官方 cell x paper-like cell 的 item count confusion matrix。"""

    matrix = pd.crosstab(validation["official_tsf_cell"], validation["paper_like_tsf_cell"])
    index = [cell for cell in TSF_CELLS if cell in matrix.index]
    columns = [cell for cell in TSF_CELLS if cell in matrix.columns]
    return matrix.reindex(index=index, columns=columns, fill_value=0)


def markdown_table_from_dataframe(df: pd.DataFrame, index_name: str) -> str:
    """生成不依赖 pandas optional `tabulate` 的 Markdown 表格。"""

    frame = df.copy()
    frame.insert(0, index_name, frame.index.astype(str))
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_validation_report(
    output_path: Path,
    channel_quality: pd.DataFrame,
    item_mean: pd.DataFrame,
    validation: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    meta: dict[str, object],
    command: str,
    tau: float,
) -> None:
    """写出 Stage 0.7 中文验证报告。"""

    lines: list[str] = []
    lines.append("# QuitoBench 通道级 full-length STL 官方 codebook 验证报告")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告用于 Stage 0.7：按论文 multivariate TSF 口径，对每个 `(subset,item_id,ind_k)` 运行 full-length Quito STL 质量评估，通道均值后用固定阈值 `tau=0.4` 构造 paper-like cell，并验证 Stage 0.6b 官方 codebook。")
    lines.append("本阶段不重新定义官方标签，不替代 Stage 0.6b codebook，不实现 router。")
    lines.append("")
    lines.append("## 2. 执行命令")
    lines.append("")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## 3. 数据来源与配置")
    lines.append("")
    lines.append("- 数据集：`hq-bench/quitobench` benchmark。")
    lines.append("- 使用 revision：`17362dcb`，原因是该版本 parquet 保留官方 `cluster` 列。")
    lines.append("- 明确未使用：`hq-bench/quito-corpus` 预训练 corpus。")
    lines.append("- 质量函数：`quito.utils.dataset_quality.evaluate_series`。")
    lines.append(f"- 二值化阈值：`tau={tau}`，规则为指标 `> tau` 是 high，`<= tau` 是 low。")
    lines.append("")
    lines.append("| subset | rows | item 数 | channel 数 | period |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for subset in ["hour", "min"]:
        if subset in meta["rows"]:
            period = 24 if subset == "hour" else 144
            lines.append(
                f"| {subset} | {meta['rows'][subset]:,} | {meta['items'][subset]:,} | "
                f"{meta['indicator_count'][subset]} | {period} |"
            )
    lines.append("")
    lines.append("## 4. 输出规模")
    lines.append("")
    lines.append(f"- 通道级质量结果：{len(channel_quality):,} 行。")
    lines.append(f"- item 级通道均值：{len(item_mean):,} 行。")
    lines.append(f"- 官方 codebook 验证表：{len(validation):,} 行。")
    lines.append("")
    lines.append("## 5. 总体验证结果")
    lines.append("")
    lines.append(f"- item exact match：{validation['item_exact_match'].mean():.2%}。")
    lines.append(f"- trend 逐维 match：{validation['trend_match'].mean():.2%}。")
    lines.append(f"- seasonality 逐维 match：{validation['seasonality_match'].mean():.2%}。")
    lines.append(f"- forecastability 逐维 match：{validation['forecastability_match'].mean():.2%}。")
    lines.append("")
    lines.append("## 6. Cluster 汇总")
    lines.append("")
    lines.append("| cluster | official cell | item 数 | paper-like 众数 | 众数占比 | exact | T match | S match | F match |")
    lines.append("| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in cluster_summary.itertuples(index=False):
        lines.append(
            f"| {row.official_cluster_code} | {row.official_tsf_cell} | {row.item_count} | "
            f"{row.paper_like_mode_cell} | {row.paper_like_mode_ratio:.2%} | "
            f"{row.item_exact_match_ratio:.2%} | {row.trend_match_ratio:.2%} | "
            f"{row.seasonality_match_ratio:.2%} | {row.forecastability_match_ratio:.2%} |"
        )
    lines.append("")
    lines.append("## 7. Confusion Matrix")
    lines.append("")
    lines.append("行是官方 cell，列是 channel-mean STL paper-like cell。")
    lines.append("")
    lines.append(markdown_table_from_dataframe(confusion_matrix, index_name="official_tsf_cell"))
    lines.append("")
    lines.append("## 8. Cluster 24 专项观察")
    lines.append("")
    cluster24 = cluster_summary[cluster_summary["official_cluster_code"] == 24]
    if not cluster24.empty:
        r = cluster24.iloc[0]
        lines.append(f"- Stage 0.6b 官方 codebook：cluster 24 -> `{r['official_tsf_cell']}`。")
        lines.append(f"- 本阶段 channel-mean STL paper-like 众数：`{r['paper_like_mode_cell']}`，众数占比 {r['paper_like_mode_ratio']:.2%}。")
        lines.append(f"- exact match：{r['item_exact_match_ratio']:.2%}；T/S/F match 分别为 {r['trend_match_ratio']:.2%}、{r['seasonality_match_ratio']:.2%}、{r['forecastability_match_ratio']:.2%}。")
        if r["paper_like_mode_cell"] == r["official_tsf_cell"] and float(r["item_exact_match_ratio"]) == 1.0:
            lines.append("- 这说明 Stage 0.6 中 cluster 24 被 item 代表序列口径解释为 `highT_highS_highF` 的冲突，在通道级 full-length STL + channel mean + `tau=0.4` 口径下已消除。cluster 24 是本阶段支持 Stage 0.6b codebook 的强证据案例。")
        else:
            lines.append("- 该结果说明 cluster 24 在本地 channel-mean STL 口径下仍存在偏差，需要把差异解释为本地 `evaluate_series` 实现、论文原始窗口/预处理或官方候选池筛选差异；不能据此推翻 Stage 0.6b codebook。")
    else:
        lines.append("- 验证表中未找到 cluster 24。")
    lines.append("")
    lines.append("## 9. 结论")
    lines.append("")
    lines.append("- Stage 0.7 已保留通道级 full-length STL 中间结果，可供后续通道独立伪图像设计复用。")
    lines.append("- 本阶段验证的是论文式 channel-mean + tau 口径与官方 codebook 的一致性，不产生新的官方标签。")
    lines.append("- 后续仍应以 Stage 0.6b codebook 作为官方 TSF cell 映射；路线 1 仍不使用 TSF 标签训练 router。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0.7 QuitoBench 通道级 STL 官方 codebook 验证。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    parser.add_argument("--codebook", type=Path, default=ROOT / "outputs/data_audit/quitobench_official_cluster_codebook.csv")
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-items-per-subset", type=int, default=None, help="仅用于 smoke test：每个 subset 最多处理多少 item。")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有通道级 CSV，从头重算。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_path = args.output_dir / "quitobench_channel_quality_stl_full.csv"
    item_mean_path = args.output_dir / "quitobench_item_quality_stl_channel_mean.csv"
    validation_path = args.output_dir / "quitobench_official_codebook_channel_stl_validation.csv"
    report_path = args.output_dir / "quitobench_official_codebook_channel_stl_validation_report.md"

    if not args.codebook.exists():
        raise FileNotFoundError(f"缺少 Stage 0.6b 官方 codebook：{args.codebook}")

    tasks, meta = iter_channel_tasks(args.data_dir, max_items_per_subset=args.max_items_per_subset)
    channel_quality = compute_channel_quality_with_progress(
        tasks=tasks,
        output_path=channel_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    item_mean = build_item_channel_mean(channel_quality, tau=args.tau)
    codebook = pd.read_csv(args.codebook)
    validation = compare_with_official_codebook(item_mean, codebook)
    cluster_summary = build_cluster_summary(validation)
    confusion_matrix = build_confusion_matrix(validation)

    item_mean.to_csv(item_mean_path, index=False)
    validation.to_csv(validation_path, index=False)
    command = (
        "conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py "
        f"--max-workers {args.max_workers} --batch-size {args.batch_size}"
    )
    if args.max_items_per_subset is not None:
        command += f" --max-items-per-subset {args.max_items_per_subset}"
    write_validation_report(
        report_path,
        channel_quality=channel_quality,
        item_mean=item_mean,
        validation=validation,
        cluster_summary=cluster_summary,
        confusion_matrix=confusion_matrix,
        meta=meta,
        command=command,
        tau=args.tau,
    )
    print(f"[done] wrote {channel_path}", flush=True)
    print(f"[done] wrote {item_mean_path}", flush=True)
    print(f"[done] wrote {validation_path}", flush=True)
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
