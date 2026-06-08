from __future__ import annotations

import pandas as pd
import pytest

from tools.quitobench_expert_cache_comparison import build_expert_comparison, build_true_uniform_ensemble_metrics


def test_build_expert_comparison_uses_common_windows_and_reports_oracle_gap() -> None:
    errors = pd.DataFrame(
        [
            {"physical_window_id": "w1", "split": "valid", "subset": "hour", "official_tsf_cell": "cell_a", "expert_id": "seasonal_naive", "mse": 4.0, "mae": 2.0},
            {"physical_window_id": "w1", "split": "valid", "subset": "hour", "official_tsf_cell": "cell_a", "expert_id": "dlinear_quito", "mse": 1.0, "mae": 1.0},
            {"physical_window_id": "w1", "split": "valid", "subset": "hour", "official_tsf_cell": "cell_a", "expert_id": "patchtst_quito", "mse": 9.0, "mae": 3.0},
            {"physical_window_id": "w2", "split": "test", "subset": "min", "official_tsf_cell": "cell_b", "expert_id": "seasonal_naive", "mse": 1.0, "mae": 1.0},
            {"physical_window_id": "w2", "split": "test", "subset": "min", "official_tsf_cell": "cell_b", "expert_id": "dlinear_quito", "mse": 4.0, "mae": 2.0},
            {"physical_window_id": "w2", "split": "test", "subset": "min", "official_tsf_cell": "cell_b", "expert_id": "patchtst_quito", "mse": 2.0, "mae": 1.5},
            {"physical_window_id": "extra", "split": "test", "subset": "min", "official_tsf_cell": "cell_b", "expert_id": "seasonal_naive", "mse": 0.0, "mae": 0.0},
        ]
    )

    summary, by_split, by_cell = build_expert_comparison(
        errors,
        required_experts=("seasonal_naive", "dlinear_quito", "patchtst_quito"),
    )

    assert summary.loc[0, "num_common_windows"] == 2
    assert summary.loc[0, "num_experts"] == 3
    assert summary.loc[0, "oracle_mse"] == pytest.approx(1.0)
    assert summary.loc[0, "best_fixed_expert"] == "dlinear_quito"
    assert summary.loc[0, "best_fixed_mse"] == pytest.approx(2.5)
    assert summary.loc[0, "oracle_gap_vs_best_fixed"] == pytest.approx(1.5)
    assert summary.loc[0, "uniform_mse_proxy"] == pytest.approx(3.5)
    assert set(by_split["split"]) == {"valid", "test"}
    assert set(by_cell["official_tsf_cell"]) == {"cell_a", "cell_b"}


def test_build_true_uniform_ensemble_metrics_averages_predictions_before_scoring() -> None:
    predictions = pd.DataFrame(
        [
            {"physical_window_id": "w1", "expert_id": "seasonal_naive", "yhat_0": 1.0, "yhat_1": 1.0},
            {"physical_window_id": "w1", "expert_id": "dlinear_quito", "yhat_0": 2.0, "yhat_1": 2.0},
            {"physical_window_id": "w1", "expert_id": "patchtst_quito", "yhat_0": 3.0, "yhat_1": 3.0},
            {"physical_window_id": "w2", "expert_id": "seasonal_naive", "yhat_0": 0.0, "yhat_1": 0.0},
            {"physical_window_id": "w2", "expert_id": "dlinear_quito", "yhat_0": 0.0, "yhat_1": 2.0},
            {"physical_window_id": "w2", "expert_id": "patchtst_quito", "yhat_0": 2.0, "yhat_1": 2.0},
            {"physical_window_id": "extra", "expert_id": "seasonal_naive", "yhat_0": 99.0, "yhat_1": 99.0},
        ]
    )
    targets = {
        "w1": [2.0, 2.0],
        "w2": [1.0, 1.0],
        "extra": [99.0, 99.0],
    }

    metrics = build_true_uniform_ensemble_metrics(
        predictions,
        targets,
        required_experts=("seasonal_naive", "dlinear_quito", "patchtst_quito"),
    )

    assert metrics.loc[0, "num_common_windows"] == 2
    assert metrics.loc[0, "num_experts"] == 3
    assert metrics.loc[0, "true_uniform_mse"] == pytest.approx(1.0 / 18.0)
    assert metrics.loc[0, "true_uniform_mae"] == pytest.approx(1.0 / 6.0)
