"""Stage 0.6：QuitoBench 官方 TSF cluster 语义画像。

本脚本不重新构造标签，不实现 router。它只读取 Stage 0.5 已固化的
官方 item 级 cluster 标签，并结合 Stage 0.1 STL 精确指标与 Stage 0
light proxy 指标，为每个官方 cluster 生成经验 TSF 语义解释。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["forecastability", "seasonality_strength", "trend_strength"]


def mode_with_ratio(values: pd.Series) -> tuple[str, float, int]:
    """返回众数、众数占比和众数计数。"""

    clean = values.dropna()
    if clean.empty:
        return "", math.nan, 0
    counts = clean.value_counts()
    mode = str(counts.index[0])
    count = int(counts.iloc[0])
    ratio = float(count / len(clean))
    return mode, ratio, count


def semantic_agreement_count(cell_a: str, cell_b: str) -> int:
    """比较两个 `highT_lowS_highF` 风格 cell 在三个维度上的一致数。"""

    parts_a = str(cell_a).split("_")
    parts_b = str(cell_b).split("_")
    if len(parts_a) != 3 or len(parts_b) != 3:
        return 0
    return sum(a == b for a, b in zip(parts_a, parts_b))


def choose_semantic_name(
    stl_mode: str,
    stl_ratio: float,
    proxy_mode: str,
    proxy_ratio: float,
) -> tuple[str, str, str]:
    """根据 STL/proxy 众数 cell 选择经验命名和置信度。

    设计原则：
    - 官方 cluster 仍是主标签，经验命名只用于解释。
    - STL 是全长 Quito 精确指标，若冲突时以 STL 众数作为建议名。
    - proxy 只作为一致性佐证或冲突提示。
    """

    if stl_mode == proxy_mode:
        if stl_ratio >= 0.6 and proxy_ratio >= 0.6:
            return stl_mode, "high", "STL 与 proxy 众数 cell 一致，且两者众数占比均不低于 0.60。"
        return stl_mode, "medium", "STL 与 proxy 众数 cell 一致，但至少一个众数占比低于 0.60。"

    agreement = semantic_agreement_count(stl_mode, proxy_mode)
    if agreement >= 2 and stl_ratio >= 0.5:
        return (
            stl_mode,
            "medium",
            f"STL 与 proxy 众数 cell 不完全一致，但有 {agreement}/3 个 TSF 维度一致；以 STL 众数作为经验命名。",
        )
    return (
        stl_mode,
        "low",
        f"STL 与 proxy 众数 cell 冲突较明显，仅 {agreement}/3 个 TSF 维度一致；保留 STL 众数作为低置信度经验命名。",
    )


def quantiles(series: pd.Series, prefix: str) -> dict[str, float]:
    """生成单个指标的分位数摘要。"""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_max": math.nan,
        }
    qs = clean.quantile([0.25, 0.5, 0.75])
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_p25": float(qs.loc[0.25]),
        f"{prefix}_median": float(qs.loc[0.5]),
        f"{prefix}_p75": float(qs.loc[0.75]),
        f"{prefix}_max": float(clean.max()),
    }


def add_item_diagnostics(final_cells: pd.DataFrame) -> pd.DataFrame:
    """为每个 item 增加 STL/proxy 是否一致等诊断字段。"""

    df = final_cells.copy()
    df["stl_proxy_agree"] = df["stl_tsf_cell"] == df["proxy_tsf_cell"]
    return df


def build_semantics(final_cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造官方 cluster 语义汇总表和 item 诊断表。"""

    diagnostics = add_item_diagnostics(final_cells)
    rows: list[dict[str, object]] = []
    for cluster_code, part in diagnostics.groupby("official_cluster_code", sort=True):
        stl_mode, stl_ratio, stl_count = mode_with_ratio(part["stl_tsf_cell"])
        proxy_mode, proxy_ratio, proxy_count = mode_with_ratio(part["proxy_tsf_cell"])
        suggested, confidence, note = choose_semantic_name(stl_mode, stl_ratio, proxy_mode, proxy_ratio)

        row: dict[str, object] = {
            "official_cluster_code": int(cluster_code),
            "official_cluster_index": int(part["official_cluster_index"].iloc[0]),
            "official_cluster_name": str(part["official_cluster_name"].iloc[0]),
            "item_count": int(len(part)),
            "hour_items": int((part["subset"] == "hour").sum()),
            "min_items": int((part["subset"] == "min").sum()),
            "stl_tsf_cell_mode": stl_mode,
            "stl_tsf_cell_mode_count": stl_count,
            "stl_tsf_cell_mode_ratio": stl_ratio,
            "proxy_tsf_cell_mode": proxy_mode,
            "proxy_tsf_cell_mode_count": proxy_count,
            "proxy_tsf_cell_mode_ratio": proxy_ratio,
            "stl_proxy_mode_agreement_dims": semantic_agreement_count(stl_mode, proxy_mode),
            "suggested_semantic_name": suggested,
            "confidence": confidence,
            "notes": note,
        }
        for metric in METRICS:
            row.update(quantiles(part[f"stl_{metric}"], f"stl_{metric}"))
            row.update(quantiles(part[f"proxy_{metric}"], f"proxy_{metric}"))
        rows.append(row)

    semantics = pd.DataFrame(rows).sort_values("official_cluster_code")
    mapping = semantics[["official_cluster_code", "suggested_semantic_name", "confidence"]]
    diagnostics = diagnostics.merge(mapping, on="official_cluster_code", how="left")
    diagnostics["cluster_semantic_match"] = diagnostics["stl_tsf_cell"] == diagnostics["suggested_semantic_name"]
    diagnostics = diagnostics[
        [
            "subset",
            "item_id",
            "official_cluster_code",
            "official_cluster_index",
            "official_cluster_name",
            "suggested_semantic_name",
            "confidence",
            "stl_tsf_cell",
            "proxy_tsf_cell",
            "stl_proxy_agree",
            "cluster_semantic_match",
            "stl_forecastability",
            "stl_seasonality_strength",
            "stl_trend_strength",
            "proxy_forecastability",
            "proxy_seasonality_strength",
            "proxy_trend_strength",
        ]
    ].sort_values(["official_cluster_code", "subset", "item_id"])
    return semantics, diagnostics


def write_report(report_path: Path, semantics: pd.DataFrame, diagnostics: pd.DataFrame, command: str) -> None:
    """写出中文 Markdown 报告。"""

    lines: list[str] = []
    lines.append("# QuitoBench 官方 TSF cluster 语义画像报告")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告解释官方 `official_cluster_code` 在 Stage 0.1 STL 精确指标和 Stage 0 light proxy 指标下的经验 TSF 含义。")
    lines.append("主标签仍为官方 cluster；本报告中的 `suggested_semantic_name` 只是经验解释，不是官方 codebook。")
    lines.append("")
    lines.append("## 2. 执行命令")
    lines.append("")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## 3. 输入输出")
    lines.append("")
    lines.append("输入：")
    lines.append("")
    lines.append("- `outputs/data_audit/quitobench_tsf_cells_final.csv`")
    lines.append("- `outputs/data_audit/quitobench_item_quality_stl.csv`")
    lines.append("- `outputs/data_audit/quitobench_item_quality.csv`")
    lines.append("")
    lines.append("输出：")
    lines.append("")
    lines.append("- `outputs/data_audit/quitobench_official_cluster_semantics.csv`")
    lines.append("- `outputs/data_audit/quitobench_official_cluster_semantics_report.md`")
    lines.append("- `outputs/data_audit/quitobench_official_cluster_item_diagnostics.csv`")
    lines.append("")
    lines.append("## 4. Cluster 语义汇总")
    lines.append("")
    lines.append("| official cluster | item 数 | STL 众数 cell | STL 众数占比 | proxy 众数 cell | proxy 众数占比 | 建议经验名 | 置信度 |")
    lines.append("| ---: | ---: | --- | ---: | --- | ---: | --- | --- |")
    for row in semantics.itertuples(index=False):
        lines.append(
            f"| {row.official_cluster_code} | {row.item_count} | {row.stl_tsf_cell_mode} | "
            f"{row.stl_tsf_cell_mode_ratio:.2%} | {row.proxy_tsf_cell_mode} | "
            f"{row.proxy_tsf_cell_mode_ratio:.2%} | {row.suggested_semantic_name} | {row.confidence} |"
        )
    lines.append("")
    lines.append("## 5. STL 指标中位数")
    lines.append("")
    lines.append("| official cluster | forecastability | seasonality | trend |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in semantics.itertuples(index=False):
        lines.append(
            f"| {row.official_cluster_code} | {row.stl_forecastability_median:.4f} | "
            f"{row.stl_seasonality_strength_median:.4f} | {row.stl_trend_strength_median:.4f} |"
        )
    lines.append("")
    lines.append("## 6. 诊断摘要")
    lines.append("")
    stl_proxy_agree = float(diagnostics["stl_proxy_agree"].mean())
    semantic_match = float(diagnostics["cluster_semantic_match"].mean())
    confidence_counts = semantics["confidence"].value_counts().to_dict()
    lines.append(f"- item 级 STL/proxy cell 完全一致率：{stl_proxy_agree:.2%}。")
    lines.append(f"- item 的 STL cell 与所属 cluster 建议经验名一致率：{semantic_match:.2%}。")
    lines.append(f"- cluster 置信度分布：{confidence_counts}。")
    lines.append("")
    lines.append("## 7. 结论")
    lines.append("")
    lines.append("- 官方 cluster 可作为路线 2 的主 TSF cell 标签。")
    lines.append("- 本报告提供每个官方 cluster 的经验 TSF 语义名和置信度，用于解释性报告。")
    lines.append("- 若某些 cluster 置信度较低，后续只应描述为“经验上接近”，不能写成官方定义。")
    lines.append("- 本阶段未执行通道级全长 STL，也未实现 router。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 QuitoBench 官方 cluster 语义画像。")
    parser.add_argument("--input-final-cells", type=Path, default=ROOT / "outputs/data_audit/quitobench_tsf_cells_final.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_cells = pd.read_csv(args.input_final_cells)
    semantics, diagnostics = build_semantics(final_cells)

    semantics_path = args.output_dir / "quitobench_official_cluster_semantics.csv"
    diagnostics_path = args.output_dir / "quitobench_official_cluster_item_diagnostics.csv"
    report_path = args.output_dir / "quitobench_official_cluster_semantics_report.md"

    semantics.to_csv(semantics_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    command = "conda run -n quito python tools/quitobench_official_cluster_semantics.py"
    write_report(report_path, semantics, diagnostics, command)

    print(f"wrote {semantics_path}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
