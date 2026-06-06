"""Stage 0.7 通道级 STL 官方 codebook 验证脚本的轻量单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from tools.quitobench_channel_stl_codebook_validation import (
    build_confusion_matrix,
    build_item_channel_mean,
    cell_from_thresholds,
    compare_with_official_codebook,
    filter_completed_channel_tasks,
    write_validation_report,
)


def test_filter_completed_channel_tasks_uses_subset_item_channel_key() -> None:
    tasks = [
        {"subset": "hour", "item_id": 1, "channel": "ind_1"},
        {"subset": "hour", "item_id": 1, "channel": "ind_2"},
        {"subset": "min", "item_id": 1, "channel": "ind_1"},
    ]
    existing = pd.DataFrame([{"subset": "hour", "item_id": 1, "channel": "ind_1"}])

    remaining = filter_completed_channel_tasks(tasks, existing)

    assert remaining == [
        {"subset": "hour", "item_id": 1, "channel": "ind_2"},
        {"subset": "min", "item_id": 1, "channel": "ind_1"},
    ]


def test_cell_from_thresholds_uses_strict_greater_than_tau() -> None:
    assert cell_from_thresholds(0.4001, 0.5, 0.6, tau=0.4) == "highT_highS_highF"
    assert cell_from_thresholds(0.4, 0.4, 0.4, tau=0.4) == "lowT_lowS_lowF"


def test_build_item_channel_mean_averages_tsf_metrics_across_channels() -> None:
    channel_quality = pd.DataFrame(
        [
            {"subset": "hour", "item_id": 1, "channel": "ind_1", "trend_strength": 0.2, "seasonality_strength": 0.6, "forecastability": 0.8},
            {"subset": "hour", "item_id": 1, "channel": "ind_2", "trend_strength": 0.6, "seasonality_strength": 0.2, "forecastability": 0.4},
            {"subset": "hour", "item_id": 2, "channel": "ind_1", "trend_strength": 0.9, "seasonality_strength": 0.9, "forecastability": 0.9},
        ]
    )

    item_mean = build_item_channel_mean(channel_quality, tau=0.4)

    row = item_mean[item_mean["item_id"] == 1].iloc[0]
    assert row["channel_count"] == 2
    assert row["complete_channel_count"] == 2
    assert row["trend_strength_channel_mean"] == 0.4
    assert row["seasonality_strength_channel_mean"] == 0.4
    assert row["forecastability_channel_mean"] == pytest.approx(0.6)
    assert row["paper_like_tsf_cell"] == "lowT_lowS_highF"


def test_compare_with_official_codebook_adds_exact_and_dimension_matches() -> None:
    item_mean = pd.DataFrame(
        [
            {"subset": "hour", "item_id": 1, "official_cluster_code": 0, "paper_like_tsf_cell": "highT_highS_highF"},
            {"subset": "hour", "item_id": 2, "official_cluster_code": 24, "paper_like_tsf_cell": "highT_highS_highF"},
        ]
    )
    codebook = pd.DataFrame(
        [
            {"official_cluster_code": 0, "official_tsf_cell": "highT_highS_highF"},
            {"official_cluster_code": 24, "official_tsf_cell": "lowT_lowS_highF"},
        ]
    )

    validation = compare_with_official_codebook(item_mean, codebook)

    row0 = validation[validation["item_id"] == 1].iloc[0]
    row24 = validation[validation["item_id"] == 2].iloc[0]
    assert bool(row0["item_exact_match"]) is True
    assert bool(row24["item_exact_match"]) is False
    assert bool(row24["trend_match"]) is False
    assert bool(row24["seasonality_match"]) is False
    assert bool(row24["forecastability_match"]) is True


def test_build_confusion_matrix_counts_official_by_paper_like_cell() -> None:
    validation = pd.DataFrame(
        [
            {"official_tsf_cell": "highT_highS_highF", "paper_like_tsf_cell": "highT_highS_highF"},
            {"official_tsf_cell": "highT_highS_highF", "paper_like_tsf_cell": "lowT_lowS_lowF"},
            {"official_tsf_cell": "lowT_lowS_lowF", "paper_like_tsf_cell": "lowT_lowS_lowF"},
        ]
    )

    matrix = build_confusion_matrix(validation)

    assert int(matrix.loc["highT_highS_highF", "highT_highS_highF"]) == 1
    assert int(matrix.loc["highT_highS_highF", "lowT_lowS_lowF"]) == 1
    assert int(matrix.loc["lowT_lowS_lowF", "lowT_lowS_lowF"]) == 1


def test_write_validation_report_does_not_require_optional_tabulate(tmp_path) -> None:
    channel_quality = pd.DataFrame([{"subset": "hour", "item_id": 1, "channel": "ind_1"}])
    item_mean = pd.DataFrame([{"subset": "hour", "item_id": 1}])
    validation = pd.DataFrame(
        [
            {
                "official_cluster_code": 0,
                "official_tsf_cell": "highT_highS_highF",
                "paper_like_tsf_cell": "highT_highS_highF",
                "item_exact_match": True,
                "trend_match": True,
                "seasonality_match": True,
                "forecastability_match": True,
            }
        ]
    )
    cluster_summary = pd.DataFrame(
        [
            {
                "official_cluster_code": 0,
                "official_tsf_cell": "highT_highS_highF",
                "item_count": 1,
                "paper_like_mode_cell": "highT_highS_highF",
                "paper_like_mode_ratio": 1.0,
                "item_exact_match_ratio": 1.0,
                "trend_match_ratio": 1.0,
                "seasonality_match_ratio": 1.0,
                "forecastability_match_ratio": 1.0,
            }
        ]
    )
    confusion = build_confusion_matrix(validation)
    report_path = tmp_path / "report.md"

    write_validation_report(
        report_path,
        channel_quality=channel_quality,
        item_mean=item_mean,
        validation=validation,
        cluster_summary=cluster_summary,
        confusion_matrix=confusion,
        meta={"rows": {"hour": 2}, "items": {"hour": 1}, "indicator_count": {"hour": 5}},
        command="python tool.py",
        tau=0.4,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "Confusion Matrix" in text
    assert "highT_highS_highF" in text
