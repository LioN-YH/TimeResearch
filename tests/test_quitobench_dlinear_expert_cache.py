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
    WindowStandardizer,
    apply_standardizer_to_series_maps,
    build_train_split_standardizer,
    build_dlinear_cache_manifest,
    build_dlinear_prediction_table,
    build_patchtst_prediction_table,
    build_tsmixer_prediction_table,
    extract_quito_standardized_series_maps,
    inverse_transform_prediction_map,
    parse_args,
    prepare_model_series_maps,
    predict_with_model,
    select_stratified_registry,
    train_quito_patchtst_model,
    train_quito_dlinear_model,
    train_quito_tsmixer_model,
    _make_model,
    _make_patchtst_model,
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


def _write_toy_quito_parquet(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2023-07-27 14:00:00", periods=14, freq="h")
    rows = []
    for item_id, offset in [(101, 0.0), (202, 100.0)]:
        for idx, dt in enumerate(dates):
            rows.append(
                {
                    "date_time": dt,
                    "item_id": item_id,
                    "cluster": 0,
                    "ind_1": offset + float(idx),
                    "ind_2": offset + 200.0 + float(idx),
                }
            )
    pd.DataFrame(rows).to_parquet(data_dir / "test_hour-00001-of-00001.parquet", index=False)


def _toy_quito_registry_for_standardization() -> pd.DataFrame:
    base = _toy_registry().iloc[0].to_dict()
    rows = []
    for window_id, item_id, channel in [("q1", 101, "ind_1"), ("q2", 202, "ind_2")]:
        row = dict(base)
        row.update(
            {
                "physical_window_id": window_id,
                "window_id": window_id,
                "subset": "hour",
                "split": "valid",
                "item_id": item_id,
                "channel": channel,
                "history_start_idx": 5,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 10,
                "history_len": 3,
                "pred_len": 2,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_registry_window_dataset_keeps_requested_split_only() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()

    dataset = RegistryWindowDataset(registry, histories, targets, split="train")

    assert len(dataset) == 2
    assert dataset.physical_window_ids == ["w_1", "w_2"]
    sample = dataset[0]
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (4, 1)


def test_build_train_split_standardizer_uses_only_train_windows() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()

    standardizer = build_train_split_standardizer(registry, histories, targets)

    train_values = np.concatenate(
        [
            histories["w_1"],
            targets["w_1"],
            histories["w_2"],
            targets["w_2"],
        ]
    )
    expected_mean = float(np.mean(train_values))
    expected_std = float(np.std(train_values) + 1e-8)
    assert standardizer.mean == pytest.approx(expected_mean)
    assert standardizer.std == pytest.approx(expected_std)
    assert standardizer.scope == "train_split_global_window_values"


def test_apply_standardizer_round_trips_histories_and_targets() -> None:
    histories, targets = _toy_histories_targets()
    standardizer = WindowStandardizer(mean=5.0, std=2.0, scope="test")

    scaled_histories, scaled_targets = apply_standardizer_to_series_maps(histories, targets, standardizer)

    assert scaled_histories["w_1"][0] == pytest.approx(-2.0)
    restored = standardizer.inverse_transform(scaled_targets["w_1"])
    np.testing.assert_allclose(restored, targets["w_1"])


def test_extract_quito_standardized_series_maps_uses_official_train_segment_per_item_channel(tmp_path: Path) -> None:
    _write_toy_quito_parquet(tmp_path)
    registry = _toy_quito_registry_for_standardization()

    histories, targets, raw_targets, scalers, summary = extract_quito_standardized_series_maps(
        registry,
        data_dir=tmp_path,
    )

    expected_std = float(np.std(np.arange(8, dtype=np.float32)) + 1e-8)
    np.testing.assert_allclose(histories["q1"], (np.array([5.0, 6.0, 7.0]) - 3.5) / expected_std)
    np.testing.assert_allclose(targets["q1"], (np.array([8.0, 9.0]) - 3.5) / expected_std)
    np.testing.assert_allclose(raw_targets["q1"], np.array([8.0, 9.0]))
    assert scalers["q1"].mean == pytest.approx(3.5)
    assert scalers["q2"].mean == pytest.approx(303.5)
    assert scalers["q1"].std == pytest.approx(expected_std)
    assert scalers["q2"].std == pytest.approx(expected_std)
    assert summary["scope"] == "quito_timeseries_dataset_train_segment"
    assert summary["scaler_granularity"] == "subset_item_channel"


def test_inverse_transform_prediction_map_uses_window_specific_quito_scalers(tmp_path: Path) -> None:
    _write_toy_quito_parquet(tmp_path)
    registry = _toy_quito_registry_for_standardization()
    _, targets, _, scalers, _ = extract_quito_standardized_series_maps(registry, data_dir=tmp_path)

    restored = inverse_transform_prediction_map(targets, scalers)

    np.testing.assert_allclose(restored["q1"], np.array([8.0, 9.0]), rtol=1e-6)
    np.testing.assert_allclose(restored["q2"], np.array([308.0, 309.0]), rtol=1e-6)


def test_prepare_model_series_maps_uses_quito_adapter_when_train_set_standardize_is_enabled(tmp_path: Path) -> None:
    _write_toy_quito_parquet(tmp_path)
    registry = _toy_quito_registry_for_standardization()

    model_histories, model_targets, error_targets, scalers, summary = prepare_model_series_maps(
        registry,
        data_dir=tmp_path,
        train_set_standardize=True,
    )

    assert scalers is not None
    assert summary["scope"] == "quito_timeseries_dataset_train_segment"
    assert model_histories["q1"].shape == (3,)
    assert model_targets["q1"].shape == (2,)
    np.testing.assert_allclose(error_targets["q1"], np.array([8.0, 9.0]))


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


def test_patchtst_config_can_disable_revin_and_override_regularization() -> None:
    config = PatchTSTExpertConfig(
        revin=False,
        dropout=0.2,
        fc_dropout=0.15,
        head_dropout=0.05,
        weight_decay=0.01,
    )

    assert config.revin is False
    assert config.dropout == pytest.approx(0.2)
    assert config.fc_dropout == pytest.approx(0.15)
    assert config.head_dropout == pytest.approx(0.05)
    assert config.weight_decay == pytest.approx(0.01)


def test_tsmixer_config_can_disable_revin_and_override_dropout() -> None:
    config = TSMixerExpertConfig(revin=False, dropout=0.2, weight_decay=0.01)

    assert config.revin is False
    assert config.dropout == pytest.approx(0.2)
    assert config.weight_decay == pytest.approx(0.01)


def test_parse_args_exposes_stage14e_alignment_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--train-set-standardize",
            "--drop-last",
            "--scheduler",
            "cosine",
            "--eta-min",
            "0.00001",
            "--scheduler-t-max",
            "50",
            "--decoder-label-len",
            "48",
            "--random-seed",
            "16",
            "--num-workers",
            "2",
            "--eval-batch-size",
            "64",
        ],
    )

    args = parse_args()

    assert args.train_set_standardize is True
    assert args.drop_last is True
    assert args.scheduler == "cosine"
    assert args.eta_min == pytest.approx(0.00001)
    assert args.scheduler_t_max == 50
    assert args.decoder_label_len == 48
    assert args.random_seed == 16
    assert args.num_workers == 2
    assert args.eval_batch_size == 64


def test_model_builders_honor_decoder_label_len() -> None:
    dlinear = _make_model(
        DLinearExpertConfig(seq_len=8, pred_len=4, decoder_label_len=4, kernel_size=3),
        device="cpu",
    )
    patchtst = _make_patchtst_model(
        PatchTSTExpertConfig(
            seq_len=8,
            pred_len=4,
            decoder_label_len=4,
            patch_len=2,
            stride=1,
            d_model=16,
            d_ff=32,
            n_heads=2,
            e_layers=1,
        ),
        device="cpu",
    )

    assert dlinear.decoder_label_len == 4
    assert dlinear.config.decoder_label_len == 4
    assert patchtst.decoder_label_len == 4
    assert patchtst.config.decoder_label_len == 4


def test_predict_with_model_inverse_transforms_standardized_predictions() -> None:
    class EchoLastModel:
        def eval(self) -> None:
            return None

        def predict(self, x, y=None):
            return x[:, -4:, :]

    registry = _toy_registry().iloc[[0]].copy()
    histories, targets = _toy_histories_targets()
    standardizer = WindowStandardizer(mean=10.0, std=2.0, scope="test")
    scaled_histories, scaled_targets = apply_standardizer_to_series_maps(histories, targets, standardizer)

    predictions = predict_with_model(
        EchoLastModel(),
        registry,
        scaled_histories,
        scaled_targets,
        config=DLinearExpertConfig(seq_len=8, pred_len=4, batch_size=1),
        device="cpu",
        output_standardizer=standardizer,
    )

    np.testing.assert_allclose(predictions["w_1"], histories["w_1"][-4:])


def test_parse_args_exposes_stage14d_diagnostic_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--expert-model",
            "patchtst",
            "--no-revin",
            "--weight-decay",
            "0.01",
            "--dropout",
            "0.2",
            "--fc-dropout",
            "0.15",
            "--head-dropout",
            "0.05",
        ],
    )

    args = parse_args()

    assert args.revin is False
    assert args.weight_decay == pytest.approx(0.01)
    assert args.dropout == pytest.approx(0.2)
    assert args.fc_dropout == pytest.approx(0.15)
    assert args.head_dropout == pytest.approx(0.05)
