"""Stage 0.6b 官方 cluster codebook 反推规则测试。"""

from __future__ import annotations

import pandas as pd

from tools.quitobench_official_cluster_codebook import (
    OFFICIAL_PAPER_REGIMES,
    PAPER_ITEM_COUNTS,
    build_candidate_rows,
    cell_from_code,
    ternary_digits,
)


def test_ternary_digits_returns_three_base3_digits() -> None:
    assert ternary_digits(0) == (0, 0, 0)
    assert ternary_digits(2) == (0, 0, 2)
    assert ternary_digits(26) == (2, 2, 2)


def test_cell_from_code_matches_paper_inferred_codebook() -> None:
    assert cell_from_code(0, digit_dims=("trend", "seasonality", "forecastability"), high_digit_by_dim={"trend": 0, "seasonality": 0, "forecastability": 0}) == "highT_highS_highF"
    assert cell_from_code(20, digit_dims=("trend", "seasonality", "forecastability"), high_digit_by_dim={"trend": 0, "seasonality": 0, "forecastability": 0}) == "lowT_highS_lowF"
    assert cell_from_code(26, digit_dims=("trend", "seasonality", "forecastability"), high_digit_by_dim={"trend": 0, "seasonality": 0, "forecastability": 0}) == "lowT_lowS_lowF"


def test_candidate_enumeration_marks_single_paper_count_match() -> None:
    rows = []
    for code, regime, item_count in zip([0, 2, 6, 8, 18, 20, 24, 26], OFFICIAL_PAPER_REGIMES, PAPER_ITEM_COUNTS):
        rows.extend(
            {"official_cluster_code": code, "stl_tsf_cell": regime, "proxy_tsf_cell": regime}
            for _ in range(item_count)
        )
    final_cells = pd.DataFrame(rows)

    candidates = build_candidate_rows(final_cells)

    assert len(candidates) == 48
    exact = candidates[candidates["paper_table_count_order_match"]]
    assert len(exact) == 1
    assert exact.iloc[0]["digit_1_dim"] == "trend"
    assert exact.iloc[0]["digit_2_dim"] == "seasonality"
    assert exact.iloc[0]["digit_3_dim"] == "forecastability"
    assert exact.iloc[0]["trend_high_digit"] == 0
    assert exact.iloc[0]["seasonality_high_digit"] == 0
    assert exact.iloc[0]["forecastability_high_digit"] == 0
