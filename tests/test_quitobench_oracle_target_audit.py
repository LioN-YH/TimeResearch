from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.quitobench_oracle_target_audit import (
    build_oracle_target_audit,
    load_cache_tables,
    write_oracle_target_audit_outputs,
)


REQUIRED_EXPERTS = ("seasonal_naive", "dlinear_quito", "patchtst_quito")


def _toy_errors() -> pd.DataFrame:
    return pd.DataFrame(
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


def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"physical_window_id": "w1", "expert_id": "seasonal_naive", "yhat_0": 0.0, "yhat_1": 0.0},
            {"physical_window_id": "w1", "expert_id": "dlinear_quito", "yhat_0": 2.0, "yhat_1": 2.0},
            {"physical_window_id": "w1", "expert_id": "patchtst_quito", "yhat_0": 4.0, "yhat_1": 4.0},
            {"physical_window_id": "w2", "expert_id": "seasonal_naive", "yhat_0": 1.0, "yhat_1": 1.0},
            {"physical_window_id": "w2", "expert_id": "dlinear_quito", "yhat_0": 3.0, "yhat_1": 3.0},
            {"physical_window_id": "w2", "expert_id": "patchtst_quito", "yhat_0": 2.0, "yhat_1": 2.0},
            {"physical_window_id": "extra", "expert_id": "seasonal_naive", "yhat_0": 99.0, "yhat_1": 99.0},
        ]
    )


def test_build_oracle_target_audit_reports_true_uniform_and_top1_rates() -> None:
    targets = {"w1": [2.0, 2.0], "w2": [1.0, 1.0], "extra": [99.0, 99.0]}

    summary, by_split, by_subset, by_cell, expert_metrics = build_oracle_target_audit(
        errors=_toy_errors(),
        predictions=_toy_predictions(),
        targets=targets,
        required_experts=REQUIRED_EXPERTS,
    )

    assert summary.loc[0, "num_common_windows"] == 2
    assert summary.loc[0, "best_fixed_expert"] == "dlinear_quito"
    assert summary.loc[0, "best_fixed_mse"] == pytest.approx(2.5)
    assert summary.loc[0, "oracle_top1_mse"] == pytest.approx(1.0)
    assert summary.loc[0, "true_uniform_mse"] == pytest.approx(0.5)
    assert summary.loc[0, "oracle_gap_vs_best_fixed"] == pytest.approx(1.5)
    assert summary.loc[0, "oracle_gap_vs_true_uniform"] == pytest.approx(-0.5)
    assert set(by_split["split"]) == {"valid", "test"}
    assert set(by_subset["subset"]) == {"hour", "min"}
    assert set(by_cell["official_tsf_cell"]) == {"cell_a", "cell_b"}
    assert expert_metrics.set_index("expert_id").loc["seasonal_naive", "oracle_top1_rate"] == pytest.approx(0.5)


def test_load_cache_tables_reads_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    cache_a = tmp_path / "a"
    cache_b = tmp_path / "b"
    cache_a.mkdir()
    cache_b.mkdir()
    _toy_predictions().query("expert_id == 'seasonal_naive'").to_parquet(cache_a / "predictions.parquet", index=False)
    _toy_errors().query("expert_id == 'seasonal_naive'").to_parquet(cache_a / "errors.parquet", index=False)
    _toy_predictions().query("expert_id == 'dlinear_quito'").to_parquet(cache_b / "predictions.parquet", index=False)
    duplicated = _toy_errors().query("expert_id == 'dlinear_quito'")
    pd.concat([duplicated, duplicated.head(1)], ignore_index=True).to_parquet(cache_b / "errors.parquet", index=False)

    with pytest.raises(ValueError, match="重复"):
        load_cache_tables([cache_a, cache_b], required_experts=("seasonal_naive", "dlinear_quito"))


def test_write_oracle_target_audit_outputs_writes_expected_files(tmp_path: Path) -> None:
    targets = {"w1": [2.0, 2.0], "w2": [1.0, 1.0]}
    tables = build_oracle_target_audit(_toy_errors(), _toy_predictions(), targets, required_experts=REQUIRED_EXPERTS)

    out_dir = write_oracle_target_audit_outputs(
        *tables,
        manifest={"audit_id": "toy"},
        output_dir=tmp_path / "audit",
    )

    assert (out_dir / "oracle_summary.csv").exists()
    assert (out_dir / "oracle_by_split.csv").exists()
    assert (out_dir / "oracle_by_subset.csv").exists()
    assert (out_dir / "oracle_by_cell.csv").exists()
    assert (out_dir / "expert_metrics.csv").exists()
    assert (out_dir / "manifest.json").exists()


def test_oracle_target_audit_cli_help_runs_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "tools/quitobench_oracle_target_audit.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--cache-dir" in result.stdout
