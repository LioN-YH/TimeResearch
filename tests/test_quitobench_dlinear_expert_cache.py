from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.quitobench_framework_expert_cache import (
    DLINEAR_EXPERT_ID,
    PATCHTST_EXPERT_ID,
    TSMIXER_EXPERT_ID,
    DLinearExpertConfig,
    PatchTSTExpertConfig,
    RegistryWindowDataset,
    TSMixerExpertConfig,
    build_dlinear_cache_manifest,
    build_dlinear_prediction_table,
    build_patchtst_prediction_table,
    build_tsmixer_prediction_table,
    select_stratified_registry,
    train_quito_patchtst_model,
    train_quito_dlinear_model,
    train_quito_tsmixer_model,
)


def test_dlinear_compatibility_module_reexports_framework_runner_symbols() -> None:
    from tools import quitobench_dlinear_expert_cache as compat

    assert compat.DLINEAR_EXPERT_ID == DLINEAR_EXPERT_ID
    assert compat.PATCHTST_EXPERT_ID == PATCHTST_EXPERT_ID
    assert compat.train_quito_dlinear_model is train_quito_dlinear_model


def _toy_registry() -> pd.DataFrame:
    rows = []
    for idx, split in enumerate(["train", "train", "valid", "test"], start=1):
        rows.append(
            {
                "physical_window_id": f"w_{idx}",
                "window_id": f"w_{idx}",
                "base_registry_id": "base_v1",
                "sample_set_id": "sample_v1",
                "subset": "hour",
                "split": split,
                "item_id": 100 + idx,
                "channel": "ind_1",
                "period": 4,
                "official_tsf_cell": "lowT_highS_highF",
                "history_start_idx": 0,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 12,
                "history_len": 8,
                "pred_len": 4,
            }
        )
    return pd.DataFrame(rows)


def _toy_histories_targets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    histories = {
        "w_1": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        "w_2": np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=float),
        "w_3": np.array([10, 10, 11, 11, 12, 12, 13, 13], dtype=float),
        "w_4": np.array([3, 1, 3, 1, 3, 1, 3, 1], dtype=float),
    }
    targets = {
        "w_1": np.array([9, 10, 11, 12], dtype=float),
        "w_2": np.array([10, 11, 12, 13], dtype=float),
        "w_3": np.array([14, 14, 15, 15], dtype=float),
        "w_4": np.array([3, 1, 3, 1], dtype=float),
    }
    return histories, targets


def test_registry_window_dataset_keeps_requested_split_only() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()

    dataset = RegistryWindowDataset(registry, histories, targets, split="train")

    assert len(dataset) == 2
    assert dataset.physical_window_ids == ["w_1", "w_2"]
    sample = dataset[0]
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (4, 1)


def test_build_dlinear_prediction_table_reuses_stage14a_schema() -> None:
    registry = _toy_registry().iloc[[2, 3]].reset_index(drop=True)
    predictions_by_id = {
        "w_3": np.array([14, 14, 15, 15], dtype=float),
        "w_4": np.array([3, 1, 3, 1], dtype=float),
    }

    table = build_dlinear_prediction_table(registry, predictions_by_id)

    assert len(table) == 2
    assert table[["physical_window_id", "expert_id"]].duplicated().sum() == 0
    assert set(table["expert_id"]) == {DLINEAR_EXPERT_ID}
    assert set(table["prediction_format"]) == {"wide_columns"}
    assert table.loc[table["physical_window_id"] == "w_3", ["yhat_0", "yhat_1", "yhat_2", "yhat_3"]].iloc[0].tolist() == [
        14.0,
        14.0,
        15.0,
        15.0,
    ]


def test_train_quito_dlinear_model_runs_tiny_training_loop() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()
    config = DLinearExpertConfig(seq_len=8, pred_len=4, epochs=1, batch_size=2, learning_rate=0.01, kernel_size=3)

    model, stats = train_quito_dlinear_model(registry, histories, targets, config=config, device="cpu")

    assert stats["train_windows"] == 2
    assert stats["trained_splits"] == ["train"]
    assert stats["epochs_completed"] == 1
    assert stats["final_train_loss"] >= 0.0
    valid_dataset = RegistryWindowDataset(registry, histories, targets, split="valid")
    sample = valid_dataset[0]
    yhat = model.predict(sample["x"].unsqueeze(0), y=None)
    assert tuple(yhat.shape) == (1, 4, 1)


def test_train_quito_patchtst_model_runs_tiny_training_loop() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()
    config = PatchTSTExpertConfig(
        seq_len=8,
        pred_len=4,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        patch_len=2,
        stride=1,
        d_model=16,
        d_ff=32,
        n_heads=2,
        e_layers=1,
    )

    model, stats = train_quito_patchtst_model(registry, histories, targets, config=config, device="cpu")

    assert stats["train_windows"] == 2
    assert stats["trained_splits"] == ["train"]
    assert stats["epochs_completed"] == 1
    assert stats["final_train_loss"] >= 0.0
    valid_dataset = RegistryWindowDataset(registry, histories, targets, split="valid")
    sample = valid_dataset[0]
    yhat = model.predict(sample["x"].unsqueeze(0), y=None)
    assert tuple(yhat.shape) == (1, 4, 1)


def test_train_quito_tsmixer_model_runs_tiny_training_loop() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()
    config = TSMixerExpertConfig(
        seq_len=8,
        pred_len=4,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        num_blocks=1,
        d_ff=8,
    )

    model, stats = train_quito_tsmixer_model(registry, histories, targets, config=config, device="cpu")

    assert stats["train_windows"] == 2
    assert stats["trained_splits"] == ["train"]
    assert stats["epochs_completed"] == 1
    assert stats["final_train_loss"] >= 0.0
    valid_dataset = RegistryWindowDataset(registry, histories, targets, split="valid")
    sample = valid_dataset[0]
    yhat = model.predict(sample["x"].unsqueeze(0), y=None)
    assert tuple(yhat.shape) == (1, 4, 1)


def test_build_patchtst_prediction_table_uses_patchtst_expert_metadata() -> None:
    registry = _toy_registry().iloc[[2, 3]].reset_index(drop=True)
    predictions_by_id = {
        "w_3": np.array([14, 14, 15, 15], dtype=float),
        "w_4": np.array([3, 1, 3, 1], dtype=float),
    }

    table = build_patchtst_prediction_table(registry, predictions_by_id)

    assert len(table) == 2
    assert table[["physical_window_id", "expert_id"]].duplicated().sum() == 0
    assert set(table["expert_id"]) == {PATCHTST_EXPERT_ID}
    assert set(table["expert_family"]) == {"patch_transformer"}


def test_build_tsmixer_prediction_table_uses_tsmixer_expert_metadata() -> None:
    registry = _toy_registry().iloc[[2, 3]].reset_index(drop=True)
    predictions_by_id = {
        "w_3": np.array([14, 14, 15, 15], dtype=float),
        "w_4": np.array([3, 1, 3, 1], dtype=float),
    }

    table = build_tsmixer_prediction_table(registry, predictions_by_id)

    assert len(table) == 2
    assert table[["physical_window_id", "expert_id"]].duplicated().sum() == 0
    assert set(table["expert_id"]) == {TSMIXER_EXPERT_ID}
    assert set(table["expert_family"]) == {"mlp_mixer"}


def test_build_dlinear_manifest_records_framework_and_no_router_flags(tmp_path: Path) -> None:
    registry = _toy_registry()
    predictions = pd.DataFrame(
        {
            "physical_window_id": ["w_3", "w_4"],
            "expert_id": [DLINEAR_EXPERT_ID, DLINEAR_EXPERT_ID],
            "split": ["valid", "test"],
        }
    )
    errors = pd.DataFrame(
        {
            "physical_window_id": ["w_3", "w_4"],
            "expert_id": [DLINEAR_EXPERT_ID, DLINEAR_EXPERT_ID],
            "mse": [1.0, 2.0],
        }
    )
    config = DLinearExpertConfig(expert_set_id="dlinear_v1__smoke", seq_len=8, pred_len=4)

    manifest = build_dlinear_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=0.5,
        input_registry_dir=tmp_path,
        max_rows=4,
        config=config,
        training_stats={"train_windows": 2, "trained_splits": ["train"]},
        audit_summary={"quito_has_dlinear": True, "tslib_available": False},
    )

    assert manifest["stage"] == "stage1_4b_dlinear_expert_cache_smoke"
    assert manifest["expert_set_id"] == "dlinear_v1__smoke"
    assert manifest["expert_ids"] == [DLINEAR_EXPERT_ID]
    assert manifest["source_framework"] == "quito"
    assert manifest["implements_router"] is False
    assert manifest["runs_visual_encoder"] is False
    assert manifest["runs_neural_experts"] is True
    assert manifest["future_read_policy"] == "history_only_for_prediction"
    assert manifest["target_usage"] == "loss_error_and_oracle_only"
    assert manifest["training_stats"]["trained_splits"] == ["train"]


def test_registry_window_dataset_rejects_non_train_training_split() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()

    with pytest.raises(ValueError, match="训练数据只能来自 train split"):
        RegistryWindowDataset(registry, histories, targets, split="valid", require_train_split=True)


def test_select_stratified_registry_balances_split_subset_cell_groups() -> None:
    rows = []
    for split in ["train", "valid", "test"]:
        for subset in ["hour", "min"]:
            for cell in ["cell_a", "cell_b"]:
                for idx in range(5):
                    base = _toy_registry().iloc[0].to_dict()
                    base.update(
                        {
                            "physical_window_id": f"{split}_{subset}_{cell}_{idx}",
                            "window_id": f"{split}_{subset}_{cell}_{idx}",
                            "split": split,
                            "subset": subset,
                            "official_tsf_cell": cell,
                            "item_id": idx,
                        }
                    )
                    rows.append(base)
    registry = pd.DataFrame(rows)

    sampled = select_stratified_registry(
        registry,
        max_rows=24,
        group_cols=("split", "subset", "official_tsf_cell"),
        random_seed=7,
    )

    assert len(sampled) == 24
    counts = sampled.groupby(["split", "subset", "official_tsf_cell"])["physical_window_id"].nunique()
    assert counts.min() == 2
    assert counts.max() == 2
    assert set(sampled["split"]) == {"train", "valid", "test"}
