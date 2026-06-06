"""Stage 0.1 item 级 STL 审计脚本的轻量单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.quitobench_item_stl_quality_audit import (
    add_cells,
    build_cell_distribution,
    filter_completed_tasks,
    representative_series,
)


def test_representative_series_zscores_channels_before_item_mean() -> None:
    item_df = pd.DataFrame(
        {
            "date_time": pd.date_range("2023-01-01", periods=4, freq="h"),
            "ind_1": [1.0, 2.0, 3.0, 4.0],
            "ind_2": [10.0, 20.0, 30.0, 40.0],
        }
    )

    actual = representative_series(item_df, ["ind_1", "ind_2"])

    expected = np.array([-1.34164077, -0.44721359, 0.44721359, 1.34164077])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_filter_completed_tasks_skips_only_matching_subset_and_item() -> None:
    tasks = [
        {"subset": "hour", "item_id": 1},
        {"subset": "hour", "item_id": 2},
        {"subset": "min", "item_id": 1},
    ]
    existing = pd.DataFrame(
        [
            {"subset": "hour", "item_id": 1, "forecastability": 0.1},
            {"subset": "min", "item_id": 1, "forecastability": 0.2},
        ]
    )

    remaining = filter_completed_tasks(tasks, existing)

    assert remaining == [{"subset": "hour", "item_id": 2}]


def test_build_cell_distribution_uses_item_windows_for_item_level() -> None:
    quality = pd.DataFrame(
        [
            {"subset": "hour", "item_id": 1, "trend_strength": 0.1, "seasonality_strength": 0.1, "forecastability": 0.1},
            {"subset": "hour", "item_id": 2, "trend_strength": 0.9, "seasonality_strength": 0.9, "forecastability": 0.9},
        ]
    )
    quality, _ = add_cells(quality)
    windows = pd.DataFrame(
        [
            {"subset": "hour", "item_id": 1, "seq_len": 96, "pred_len": 96, "split": "train", "item_windows": 3},
            {"subset": "hour", "item_id": 1, "seq_len": 96, "pred_len": 96, "split": "valid", "item_windows": 2},
            {"subset": "hour", "item_id": 1, "seq_len": 96, "pred_len": 96, "split": "test", "item_windows": 1},
            {"subset": "hour", "item_id": 2, "seq_len": 96, "pred_len": 96, "split": "train", "item_windows": 30},
            {"subset": "hour", "item_id": 2, "seq_len": 96, "pred_len": 96, "split": "valid", "item_windows": 20},
            {"subset": "hour", "item_id": 2, "seq_len": 96, "pred_len": 96, "split": "test", "item_windows": 10},
        ]
    )

    dist = build_cell_distribution(quality, windows, window_settings=[(96, 96)])

    low_cell = dist[dist["tsf_cell"] == "lowT_lowS_lowF"].iloc[0]
    high_cell = dist[dist["tsf_cell"] == "highT_highS_highF"].iloc[0]
    assert int(low_cell["unit_count"]) == 1
    assert int(low_cell["train_windows"]) == 3
    assert int(low_cell["valid_windows"]) == 2
    assert int(low_cell["test_windows"]) == 1
    assert int(high_cell["train_windows"]) == 30
