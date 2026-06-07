from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.quitobench_lightweight_expert_cache import (
    EXPERT_IDS,
    REQUIRED_REGISTRY_COLUMNS,
    build_cache_manifest,
    compute_error_table,
    compute_lightweight_expert_predictions,
    compute_oracle_summary,
    select_stratified_registry,
    validate_registry,
    write_expert_cache_outputs,
)


def _toy_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "physical_window_id": "w_1",
                "window_id": "w_1",
                "base_registry_id": "base_v1",
                "sample_set_id": "sample_v1",
                "subset": "hour",
                "split": "valid",
                "item_id": "item_a",
                "channel": "ind_1",
                "period": 4,
                "official_tsf_cell": "lowT_highS_highF",
                "history_start_idx": 0,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 12,
                "history_len": 8,
                "pred_len": 4,
            },
            {
                "physical_window_id": "w_2",
                "window_id": "w_2",
                "base_registry_id": "base_v1",
                "sample_set_id": "sample_v1",
                "subset": "hour",
                "split": "valid",
                "item_id": "item_b",
                "channel": "ind_1",
                "period": 4,
                "official_tsf_cell": "highT_lowS_highF",
                "history_start_idx": 0,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 12,
                "history_len": 8,
                "pred_len": 4,
            },
        ]
    )


def test_required_registry_columns_include_stage1_keys() -> None:
    assert "physical_window_id" in REQUIRED_REGISTRY_COLUMNS
    assert "sample_set_id" in REQUIRED_REGISTRY_COLUMNS
    assert "target_start_idx" in REQUIRED_REGISTRY_COLUMNS
    assert "pred_len" in REQUIRED_REGISTRY_COLUMNS


def test_compute_lightweight_expert_predictions_uses_history_only() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)

    assert set(predictions["expert_id"]) == set(EXPERT_IDS)
    assert len(predictions) == 4
    assert predictions[["physical_window_id", "expert_id"]].duplicated().sum() == 0

    wide_cols = ["yhat_0", "yhat_1", "yhat_2", "yhat_3"]
    by_expert = predictions.set_index("expert_id")

    assert by_expert.loc["last_value", wide_cols].to_numpy(dtype=float).tolist() == [40.0, 40.0, 40.0, 40.0]
    assert by_expert.loc["seasonal_naive", wide_cols].to_numpy(dtype=float).tolist() == [10.0, 20.0, 30.0, 40.0]
    assert by_expert.loc["recent_mean", wide_cols].to_numpy(dtype=float).tolist() == [35.0, 35.0, 35.0, 35.0]

    linear = by_expert.loc["linear_trend", wide_cols].to_numpy(dtype=float)
    np.testing.assert_allclose(linear, np.array([38.92857143, 44.52380952, 50.11904762, 55.71428571]), atol=1e-8)


def test_compute_lightweight_expert_predictions_can_select_single_expert() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}

    predictions = compute_lightweight_expert_predictions(registry, histories, expert_ids=("seasonal_naive",))

    assert predictions["expert_id"].tolist() == ["seasonal_naive"]
    assert predictions[["yhat_0", "yhat_1", "yhat_2", "yhat_3"]].iloc[0].to_numpy(dtype=float).tolist() == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]


def test_compute_error_table_and_oracle_summary() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    targets = {"w_1": np.array([10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)

    errors = compute_error_table(predictions, targets)

    assert set(errors["expert_id"]) == set(EXPERT_IDS)
    seasonal = errors.set_index("expert_id").loc["seasonal_naive"]
    assert seasonal["mse"] == pytest.approx(0.0)
    assert seasonal["mae"] == pytest.approx(0.0)
    assert bool(seasonal["is_oracle_top1"]) is True

    weights = errors.groupby("physical_window_id")["soft_oracle_weight"].sum()
    assert weights.loc["w_1"] == pytest.approx(1.0)

    summary = compute_oracle_summary(errors)
    assert summary.loc[0, "num_windows"] == 1
    assert summary.loc[0, "oracle_mse"] == pytest.approx(0.0)
    assert summary.loc[0, "best_fixed_expert"] == "seasonal_naive"


def test_validate_registry_rejects_duplicate_physical_window_id() -> None:
    registry = pd.concat([_toy_registry().iloc[[0]], _toy_registry().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="physical_window_id 不唯一"):
        validate_registry(registry)


def test_build_cache_manifest_rejects_expert_id_mismatch() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    targets = {"w_1": np.array([10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)
    errors = compute_error_table(predictions, targets)

    with pytest.raises(ValueError, match="expert_ids 与 predictions/errors 实际专家集合不一致"):
        build_cache_manifest(
            registry=registry,
            predictions=predictions,
            errors=errors,
            elapsed_seconds=0.5,
            input_registry_dir=Path("/tmp/registry"),
            max_rows=1,
            stratified_rows=None,
            stratify_columns=("split", "subset", "official_tsf_cell"),
            expert_ids=("seasonal_naive",),
        )


def test_select_stratified_registry_balances_split_subset_cell() -> None:
    rows = []
    for split in ["valid", "test"]:
        for subset in ["hour", "min"]:
            for cell in ["cell_a", "cell_b"]:
                for idx in range(5):
                    rows.append(
                        {
                            **_toy_registry().iloc[0].to_dict(),
                            "physical_window_id": f"{split}_{subset}_{cell}_{idx}",
                            "window_id": f"{split}_{subset}_{cell}_{idx}",
                            "split": split,
                            "subset": subset,
                            "official_tsf_cell": cell,
                        }
                    )
    registry = pd.DataFrame(rows)

    sampled = select_stratified_registry(
        registry,
        target_rows=16,
        stratify_columns=("split", "subset", "official_tsf_cell"),
        random_seed=7,
    )

    assert len(sampled) == 16
    assert sampled["physical_window_id"].is_unique
    group_counts = sampled.groupby(["split", "subset", "official_tsf_cell"]).size()
    assert group_counts.nunique() == 1
    assert int(group_counts.iloc[0]) == 2


def test_write_expert_cache_outputs_writes_expected_files(tmp_path: Path) -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    targets = {"w_1": np.array([10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories, expert_ids=("seasonal_naive",))
    errors = compute_error_table(predictions, targets)
    oracle = compute_oracle_summary(errors)
    cell_matrix = errors.groupby(["official_tsf_cell", "expert_id"], as_index=False)["mse"].mean()
    manifest = build_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=0.5,
        input_registry_dir=Path("/tmp/registry"),
        max_rows=1,
        stratified_rows=None,
        stratify_columns=("split", "subset", "official_tsf_cell"),
        expert_ids=("seasonal_naive",),
    )

    out_dir = write_expert_cache_outputs(
        predictions=predictions,
        errors=errors,
        oracle_summary=oracle,
        cell_model_matrix=cell_matrix,
        manifest=manifest,
        output_root=tmp_path,
        expert_set_id="lightweight_v1",
    )

    assert (out_dir / "predictions.parquet").exists()
    assert (out_dir / "errors.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "profiling/cell_model_matrix.csv").exists()
    assert (out_dir / "profiling/oracle_summary.csv").exists()

    loaded_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["stage"] == "stage1_4a_lightweight_expert_cache"
    assert loaded_manifest["expert_set_id"] == "lightweight_v1"
    assert loaded_manifest["expert_ids"] == ["seasonal_naive"]
    assert loaded_manifest["implements_router"] is False
    assert loaded_manifest["runs_visual_encoder"] is False
    assert loaded_manifest["runs_neural_experts"] is False
    assert loaded_manifest["future_read_policy"] == "history_only_for_prediction"

    loaded_predictions = pd.read_parquet(out_dir / "predictions.parquet")
    assert loaded_predictions[["physical_window_id", "expert_id"]].duplicated().sum() == 0
