from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tools.quitobench_expert_cache_audit import audit_cache_collection, audit_single_cache, write_cache_audit_outputs


def _write_cache(cache_dir: Path, expert_id: str, window_ids: list[str]) -> Path:
    cache_dir.mkdir(parents=True)
    predictions = pd.DataFrame(
        [
            {
                "physical_window_id": window_id,
                "expert_id": expert_id,
                "split": "train" if window_id == "w1" else "test",
                "subset": "hour",
                "official_tsf_cell": "cell_a",
                "yhat_0": 1.0,
                "yhat_1": 2.0,
            }
            for window_id in window_ids
        ]
    )
    errors = predictions[["physical_window_id", "expert_id", "split", "subset", "official_tsf_cell"]].copy()
    errors["mse"] = 1.0
    errors["mae"] = 0.5
    manifest = {
        "expert_set_id": cache_dir.name,
        "expert_ids": [expert_id],
        "sample_set_id": ["sample_a"],
        "base_registry_id": ["base_a"],
        "total_windows": len(window_ids),
        "prediction_rows": len(window_ids),
        "error_rows": len(window_ids),
        "config": {"pred_len": 2, "train_set_standardize": True},
        "standardization": {"enabled": True, "scope": "quito_timeseries_dataset_train_segment"},
    }
    predictions.to_parquet(cache_dir / "predictions.parquet", index=False)
    errors.to_parquet(cache_dir / "errors.parquet", index=False)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache_dir


def test_audit_single_cache_reports_key_horizon_and_manifest_fields(tmp_path: Path) -> None:
    cache_dir = _write_cache(tmp_path / "dlinear_cache", "dlinear_quito", ["w1", "w2"])

    summary = audit_single_cache(cache_dir)

    assert summary["cache_dir"] == str(cache_dir)
    assert summary["expert_ids"] == ["dlinear_quito"]
    assert summary["prediction_rows"] == 2
    assert summary["error_rows"] == 2
    assert summary["unique_prediction_key"] is True
    assert summary["unique_error_key"] is True
    assert summary["num_yhat_cols"] == 2
    assert summary["manifest_pred_len"] == 2
    assert summary["horizon_matches_manifest"] is True
    assert summary["standardization_enabled"] is True


def test_audit_cache_collection_reports_common_window_intersection(tmp_path: Path) -> None:
    cache_a = _write_cache(tmp_path / "snaive_cache", "seasonal_naive", ["w1", "w2", "w3"])
    cache_b = _write_cache(tmp_path / "dlinear_cache", "dlinear_quito", ["w2", "w3", "w4"])

    summary, per_cache = audit_cache_collection([cache_a, cache_b])

    assert summary["num_caches"] == 2
    assert summary["expert_ids"] == ["dlinear_quito", "seasonal_naive"]
    assert summary["common_prediction_windows"] == 2
    assert summary["common_error_windows"] == 2
    assert per_cache["prediction_rows"].tolist() == [3, 3]


def test_write_cache_audit_outputs_writes_summary_and_per_cache_table(tmp_path: Path) -> None:
    cache_dir = _write_cache(tmp_path / "snaive_cache", "seasonal_naive", ["w1"])
    summary, per_cache = audit_cache_collection([cache_dir])

    out_dir = write_cache_audit_outputs(summary, per_cache, tmp_path / "audit")

    assert (out_dir / "cache_audit_summary.json").exists()
    assert (out_dir / "cache_audit_per_cache.csv").exists()


def test_cli_help_runs_when_invoked_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "tools/quitobench_expert_cache_audit.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--cache-dir" in result.stdout
