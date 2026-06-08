from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.quitobench_framework_expert_cache import QuitoWindowScaler
from tools.quitobench_normalized_oracle_audit import build_normalized_prediction_error_tables


def test_build_normalized_prediction_error_tables_transforms_predictions_and_scores_errors() -> None:
    predictions = pd.DataFrame(
        [
            {
                "physical_window_id": "w1",
                "expert_id": "expert_a",
                "split": "test",
                "subset": "hour",
                "official_tsf_cell": "cell_a",
                "yhat_0": 12.0,
                "yhat_1": 14.0,
            },
            {
                "physical_window_id": "w1",
                "expert_id": "expert_b",
                "split": "test",
                "subset": "hour",
                "official_tsf_cell": "cell_a",
                "yhat_0": 10.0,
                "yhat_1": 18.0,
            },
            {
                "physical_window_id": "w2",
                "expert_id": "expert_a",
                "split": "valid",
                "subset": "min",
                "official_tsf_cell": "cell_b",
                "yhat_0": 101.0,
                "yhat_1": 99.0,
            },
        ]
    )
    normalized_targets = {
        "w1": np.asarray([1.0, 2.0], dtype=np.float32),
        "w2": np.asarray([0.5, -0.5], dtype=np.float32),
    }
    scalers = {
        "w1": QuitoWindowScaler(mean=10.0, std=2.0, subset="hour", item_id=1, channel="ind_1"),
        "w2": QuitoWindowScaler(mean=100.0, std=2.0, subset="min", item_id=2, channel="ind_1"),
    }

    normalized_predictions, normalized_errors = build_normalized_prediction_error_tables(
        predictions,
        normalized_targets=normalized_targets,
        scalers_by_id=scalers,
    )

    yhat_cols = ["yhat_0", "yhat_1"]
    assert normalized_predictions.loc[0, yhat_cols].tolist() == [1.0, 2.0]
    assert normalized_predictions.loc[1, yhat_cols].tolist() == [0.0, 4.0]
    assert normalized_predictions.loc[2, yhat_cols].tolist() == [0.5, -0.5]
    assert normalized_errors.loc[0, "mse"] == 0.0
    assert normalized_errors.loc[0, "mae"] == 0.0
    assert normalized_errors.loc[1, "mse"] == 2.5
    assert normalized_errors.loc[1, "mae"] == 1.5
    assert normalized_errors.loc[2, "mse"] == 0.0


def test_cli_help_runs_when_invoked_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "tools/quitobench_normalized_oracle_audit.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--cache-dir" in result.stdout
