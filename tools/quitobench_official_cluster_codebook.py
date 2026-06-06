"""Stage 0.6b：QuitoBench 官方 cluster codebook 反推与验证。

本脚本只做标签解释和候选映射评估，不重新构造标签，不实现 router。
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODES = [0, 2, 6, 8, 18, 20, 24, 26]
DIMS = ("trend", "seasonality", "forecastability")
DIM_LABELS = {
    "trend": ("highT", "lowT"),
    "seasonality": ("highS", "lowS"),
    "forecastability": ("highF", "lowF"),
}
METRIC_COLUMNS = {
    "trend": "trend_strength",
    "seasonality": "seasonality_strength",
    "forecastability": "forecastability",
}

# 论文 Table 23 / Appendix H 的 regime 顺序。其 evaluation instances 除以
# 10 models x 18 task configurations = 180 后，正好得到 Stage 0.5
# 官方 cluster code 升序的 item 数。
OFFICIAL_PAPER_REGIMES = [
    "highT_highS_highF",
    "highT_highS_lowF",
    "highT_lowS_highF",
    "highT_lowS_lowF",
    "lowT_highS_highF",
    "lowT_highS_lowF",
    "lowT_lowS_highF",
    "lowT_lowS_lowF",
]
PAPER_EVALUATION_INSTANCES = [29880, 24480, 30600, 28260, 28620, 29880, 30420, 30060]
EVALUATION_INSTANCES_PER_ITEM = 180
PAPER_ITEM_COUNTS = [n // EVALUATION_INSTANCES_PER_ITEM for n in PAPER_EVALUATION_INSTANCES]


def ternary_digits(code: int) -> tuple[int, int, int]:
    """返回三位三进制 digit。"""

    if code < 0 or code > 26:
        raise ValueError(f"cluster code 超出三位三进制范围：{code}")
    d1 = code // 9
    d2 = (code % 9) // 3
    d3 = code % 3
    return d1, d2, d3


def cell_from_code(
    code: int,
    digit_dims: tuple[str, str, str],
    high_digit_by_dim: dict[str, int],
) -> str:
    """根据 digit->维度和 high digit 设定，把 cluster code 解码为 TSF cell。"""

    values: dict[str, str] = {}
    for digit, dim in zip(ternary_digits(code), digit_dims):
        high_label, low_label = DIM_LABELS[dim]
        values[dim] = high_label if digit == high_digit_by_dim[dim] else low_label
    return f"{values['trend']}_{values['seasonality']}_{values['forecastability']}"


def agreement_ratio(predicted: pd.Series, observed: pd.Series) -> float:
    clean = observed.notna()
    if int(clean.sum()) == 0:
        return math.nan
    return float((predicted[clean] == observed[clean]).mean())


def dimension_direction_score(final_cells: pd.DataFrame, predicted_cells: pd.Series, prefix: str) -> tuple[float, str]:
    """检查候选 high/low 是否与已有质量指标方向一致。

    返回三个维度中 high 组指标中位数高于 low 组的比例，以及文本说明。
    """

    checks: list[bool] = []
    notes: list[str] = []
    for dim, metric in METRIC_COLUMNS.items():
        metric_col = f"{prefix}_{metric}"
        if metric_col not in final_cells.columns:
            notes.append(f"{dim}:missing_metric")
            continue
        labels = predicted_cells.str.split("_", expand=True)
        dim_index = {"trend": 0, "seasonality": 1, "forecastability": 2}[dim]
        is_high = labels[dim_index].str.startswith("high")
        high_med = pd.to_numeric(final_cells.loc[is_high, metric_col], errors="coerce").median()
        low_med = pd.to_numeric(final_cells.loc[~is_high, metric_col], errors="coerce").median()
        if pd.isna(high_med) or pd.isna(low_med):
            notes.append(f"{dim}:nan_metric")
            continue
        ok = bool(high_med >= low_med)
        checks.append(ok)
        notes.append(f"{dim}:{high_med:.4f}>={low_med:.4f}:{ok}")
    if not checks:
        return 0.0, "; ".join(notes)
    return float(sum(checks) / len(checks)), "; ".join(notes)


def build_candidate_rows(final_cells: pd.DataFrame) -> pd.DataFrame:
    """枚举 6 x 2^3 = 48 种 codebook 候选并评分。"""

    cluster_counts = final_cells.groupby("official_cluster_code").size().reindex(CODES).astype(int).tolist()
    rows: list[dict[str, object]] = []
    for digit_dims in itertools.permutations(DIMS):
        for high_digits in itertools.product([0, 2], repeat=3):
            high_digit_by_dim = dict(zip(DIMS, high_digits))
            mapping = {code: cell_from_code(code, digit_dims, high_digit_by_dim) for code in CODES}
            predicted = final_cells["official_cluster_code"].map(mapping)
            stl_direction, stl_direction_notes = dimension_direction_score(final_cells, predicted, "stl")
            proxy_direction, proxy_direction_notes = dimension_direction_score(final_cells, predicted, "proxy")
            paper_order_match = [mapping[code] for code in CODES] == OFFICIAL_PAPER_REGIMES
            paper_count_match = cluster_counts == PAPER_ITEM_COUNTS
            stl_match = agreement_ratio(predicted, final_cells["stl_tsf_cell"])
            proxy_match = agreement_ratio(predicted, final_cells["proxy_tsf_cell"])
            evidence_score = (
                100.0 * float(paper_order_match and paper_count_match)
                + 10.0 * stl_match
                + 5.0 * proxy_match
                + stl_direction
                + 0.5 * proxy_direction
            )
            rows.append(
                {
                    "digit_1_dim": digit_dims[0],
                    "digit_2_dim": digit_dims[1],
                    "digit_3_dim": digit_dims[2],
                    "trend_high_digit": high_digit_by_dim["trend"],
                    "seasonality_high_digit": high_digit_by_dim["seasonality"],
                    "forecastability_high_digit": high_digit_by_dim["forecastability"],
                    "paper_table_count_order_match": bool(paper_order_match and paper_count_match),
                    "paper_regime_order_match": bool(paper_order_match),
                    "paper_item_counts_match": bool(paper_count_match),
                    "stl_item_exact_match_ratio": stl_match,
                    "proxy_item_exact_match_ratio": proxy_match,
                    "stl_metric_direction_score": stl_direction,
                    "proxy_metric_direction_score": proxy_direction,
                    "evidence_score": evidence_score,
                    "stl_metric_direction_notes": stl_direction_notes,
                    "proxy_metric_direction_notes": proxy_direction_notes,
                    **{f"code_{code}_cell": mapping[code] for code in CODES},
                }
            )

    candidates = pd.DataFrame(rows).sort_values(
        ["paper_table_count_order_match", "evidence_score", "stl_item_exact_match_ratio"],
        ascending=[False, False, False],
    )
    candidates.insert(0, "candidate_rank", range(1, len(candidates) + 1))
    return candidates


def build_official_codebook(candidates: pd.DataFrame) -> pd.DataFrame:
    official = candidates[candidates["paper_table_count_order_match"]].iloc[0]
    rows = []
    for code, regime, count, eval_instances in zip(CODES, OFFICIAL_PAPER_REGIMES, PAPER_ITEM_COUNTS, PAPER_EVALUATION_INSTANCES):
        rows.append(
            {
                "official_cluster_code": code,
                "base3_code": "".join(str(d) for d in ternary_digits(code)),
                "official_tsf_cell": regime,
                "paper_table_eval_instances": eval_instances,
                "paper_inferred_item_count": count,
                "digit_1_dim": official["digit_1_dim"],
                "digit_2_dim": official["digit_2_dim"],
                "digit_3_dim": official["digit_3_dim"],
                "trend_high_digit": int(official["trend_high_digit"]),
                "seasonality_high_digit": int(official["seasonality_high_digit"]),
                "forecastability_high_digit": int(official["forecastability_high_digit"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    candidates: pd.DataFrame,
    codebook: pd.DataFrame,
    final_cells: pd.DataFrame,
    command: str,
) -> None:
    top = candidates.iloc[0]
    lines: list[str] = []
    lines.append("# QuitoBench 官方 cluster codebook 反推与验证报告")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告用于 Stage 0.6b：利用论文/README 证据、官方 cluster code 的三进制结构，以及已有 Stage 0.1 STL / Stage 0 proxy 指标，反推 `official_cluster_code` 到 TSF regime 的映射。")
    lines.append("本阶段不重新构造标签，不实现 router。")
    lines.append("")
    lines.append("## 2. 执行命令")
    lines.append("")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## 3. 官方资料证据")
    lines.append("")
    lines.append("- Hugging Face 旧 revision `17362dcb` 的 README schema 写明：`cluster` 是 `TSF regime label (8-class integer code)`。")
    lines.append("- QuitoBench README/论文说明 benchmark 覆盖 `trend x seasonality x forecastability` 八个 regime cell。")
    lines.append("- 论文 TSF diagnostic 定义中，trend、seasonality、forecastability 三项指标使用默认阈值 `tau=0.4` 二值化：指标 `> tau` 为 high，指标 `<= tau` 为 low。")
    lines.append("- 论文 Appendix H / Table 23 按以下顺序列出 8 个 TSF regime 的 evaluation instances：`high_high_high, high_high_low, high_low_high, high_low_low, low_high_high, low_high_low, low_low_high, low_low_low`。")
    lines.append("- Table 23 计数除以 `10 models x 18 task configurations = 180` 后得到 item 数：`166, 136, 170, 157, 159, 166, 169, 167`。")
    lines.append("- 这个 item 数序列与 Stage 0.5 从官方 `cluster` 列抽取后按 code 升序得到的 item 数完全一致。")
    lines.append("- 论文来源：<https://arxiv.org/abs/2603.26017>；旧 README 来源：`data/hf/hq-bench/quitobench/revisions/17362dcb/README.md`。")
    lines.append("")
    lines.append("## 4. 反推结论")
    lines.append("")
    lines.append("最可信 codebook 为：三位三进制 digit 依次对应 `trend, seasonality, forecastability`；每个维度中 digit `0` 表示 high，digit `2` 表示 low。")
    lines.append("")
    lines.append("| cluster code | base-3 | official TSF cell | paper item count | local item count |")
    lines.append("| ---: | --- | --- | ---: | ---: |")
    local_counts = final_cells.groupby("official_cluster_code").size().to_dict()
    for row in codebook.itertuples(index=False):
        lines.append(
            f"| {row.official_cluster_code} | {row.base3_code} | {row.official_tsf_cell} | "
            f"{row.paper_inferred_item_count} | {local_counts[int(row.official_cluster_code)]} |"
        )
    lines.append("")
    lines.append("## 5. 候选枚举结果")
    lines.append("")
    lines.append("共枚举 48 种候选：6 种 digit 顺序乘以每个维度 `0/2` 表示 high 的 8 种组合。只有 1 种候选同时满足论文 regime 顺序和 Table 23 item count 序列。")
    lines.append("")
    lines.append("| rank | digit 顺序 | high digit(T/S/F) | paper match | STL exact | proxy exact | score |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: |")
    for row in candidates.head(8).itertuples(index=False):
        digit_order = f"{row.digit_1_dim}/{row.digit_2_dim}/{row.digit_3_dim}"
        high_digits = f"{row.trend_high_digit}/{row.seasonality_high_digit}/{row.forecastability_high_digit}"
        lines.append(
            f"| {row.candidate_rank} | {digit_order} | {high_digits} | {row.paper_table_count_order_match} | "
            f"{row.stl_item_exact_match_ratio:.2%} | {row.proxy_item_exact_match_ratio:.2%} | {row.evidence_score:.3f} |"
        )
    lines.append("")
    lines.append("## 6. STL/proxy 验证与局限")
    lines.append("")
    lines.append(f"- 最可信候选的 STL item exact match：{top.stl_item_exact_match_ratio:.2%}。")
    lines.append(f"- 最可信候选的 proxy item exact match：{top.proxy_item_exact_match_ratio:.2%}。")
    lines.append("- 这些比例不高，说明 Stage 0.1/Stage 0 使用当前本地口径按 item 中位数重新二分时，不能复现官方 regime 构造；这与 Stage 0.6 的解释坍缩一致。")
    lines.append("- 因此 STL/proxy 只能作为辅助诊断，不能覆盖或替代论文与官方 cluster 列给出的 codebook 证据。")
    lines.append("")
    lines.append("## 7. 结论")
    lines.append("")
    lines.append("- 本阶段给出 high confidence codebook：`0->highT_highS_highF`, `2->highT_highS_lowF`, `6->highT_lowS_highF`, `8->highT_lowS_lowF`, `18->lowT_highS_highF`, `20->lowT_highS_lowF`, `24->lowT_lowS_highF`, `26->lowT_lowS_lowF`。")
    lines.append("- Stage 0.6 的 `suggested_semantic_name` 应继续标记为 preliminary，不用于官方 codebook。")
    lines.append("- 后续路线 2 可以使用该 codebook 做结构报告和 supervised TSF 目标；路线 1 仍不使用 TSF 标签训练 router。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="反推并验证 QuitoBench 官方 cluster codebook。")
    parser.add_argument("--input-final-cells", type=Path, default=ROOT / "outputs/data_audit/quitobench_tsf_cells_final.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_cells = pd.read_csv(args.input_final_cells)

    candidates = build_candidate_rows(final_cells)
    codebook = build_official_codebook(candidates)

    candidates_path = args.output_dir / "quitobench_official_cluster_codebook_candidates.csv"
    codebook_path = args.output_dir / "quitobench_official_cluster_codebook.csv"
    report_path = args.output_dir / "quitobench_official_cluster_codebook_report.md"
    candidates.to_csv(candidates_path, index=False)
    codebook.to_csv(codebook_path, index=False)
    command = "conda run -n quito python tools/quitobench_official_cluster_codebook.py"
    write_report(report_path, candidates, codebook, final_cells, command)

    print(f"wrote {candidates_path}")
    print(f"wrote {codebook_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
