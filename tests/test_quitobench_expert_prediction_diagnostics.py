from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.quitobench_expert_prediction_diagnostics import summarize_prediction_scale


def test_summarize_prediction_scale_reports_prediction_target_and_error_stats() -> None:
    predictions = pd.DataFrame(
        {
            "physical_window_id": ["w1", "w2"],
            "expert_id": ["patchtst_quito", "patchtst_quito"],
            "yhat_0": [1.0, 10.0],
            "yhat_1": [2.0, 20.0],
        }
    )
    targets = {
        "w1": np.array([1.5, 2.5], dtype=float),
        "w2": np.array([9.0, 19.0], dtype=float),
    }

    summary = summarize_prediction_scale(predictions, targets)

    assert summary["rows"] == 2
    assert summary["horizon_columns"] == 2
    assert summary["prediction"]["max"] == pytest.approx(20.0)
    assert summary["target"]["min"] == pytest.approx(1.5)
    assert summary["absolute_error"]["max"] == pytest.approx(1.0)
    assert summary["finite_prediction_rate"] == pytest.approx(1.0)


def test_diagnostics_script_help_runs_when_invoked_by_path() -> None:
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "tools/quitobench_expert_prediction_diagnostics.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--cache-dir" in result.stdout
