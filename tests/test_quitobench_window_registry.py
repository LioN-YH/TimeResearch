from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.quitobench_window_registry import (
    RegistryConfig,
    build_config_hash,
    build_window_registry,
    build_sample_set_id,
    iter_window_offsets,
    load_official_codebook,
    split_bounds_from_dates,
    write_registry_outputs,
)


def _toy_df() -> pd.DataFrame:
    dates = pd.date_range("2023-07-27 14:00:00", periods=14, freq="h")
    rows = []
    for item_id, cluster in [(101, 24), (202, 0)]:
        for idx, dt in enumerate(dates):
            rows.append(
                {
                    "date_time": dt,
                    "item_id": item_id,
                    "cluster": cluster,
                    "ind_1": float(idx),
                    "ind_2": float(idx + 100),
                }
            )
    return pd.DataFrame(rows)


def test_default_config_uses_working_stride() -> None:
    assert RegistryConfig().sample_stride == 96
    assert RegistryConfig().split_context_policy == "quito_overlap"


def test_config_hash_is_stable_and_order_independent() -> None:
    cfg_a = RegistryConfig(history_len=3, pred_len=2, sample_stride=2, channels=("ind_1", "ind_2"))
    cfg_b = RegistryConfig(channels=("ind_1", "ind_2"), sample_stride=2, pred_len=2, history_len=3)

    assert build_config_hash(cfg_a) == build_config_hash(cfg_b)
    assert len(build_config_hash(cfg_a)) == 12


def test_sample_set_id_changes_with_sampling_policy() -> None:
    cfg_a = RegistryConfig(history_len=3, pred_len=2, sample_stride=1)
    cfg_b = RegistryConfig(history_len=3, pred_len=2, sample_stride=2)

    assert build_sample_set_id(cfg_a) != build_sample_set_id(cfg_b)


def test_split_bounds_from_dates_uses_quito_temporal_cutoff() -> None:
    dates = pd.date_range("2023-07-27 14:00:00", periods=14, freq="h")

    bounds = split_bounds_from_dates(dates)

    assert bounds["train"] == (0, 8)
    assert bounds["valid"] == (8, 10)
    assert bounds["test"] == (10, 14)


def test_build_window_registry_creates_channel_independent_rows() -> None:
    codebook = pd.DataFrame(
        {
            "official_cluster_code": [0, 24],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_highF"],
        }
    )
    cfg = RegistryConfig(
        history_len=3,
        pred_len=2,
        sample_stride=2,
        split_context_policy="strict_within_split",
        channels=("ind_1", "ind_2"),
    )

    registry, manifest = build_window_registry({"hour": _toy_df()}, codebook, cfg)

    assert manifest["total_windows"] == len(registry)
    assert set(registry["subset"]) == {"hour"}
    assert set(registry["channel"]) == {"ind_1", "ind_2"}
    assert set(registry["official_tsf_cell"]) == {"highT_highS_highF", "lowT_lowS_highF"}
    assert registry["window_id"].is_unique
    assert registry["config_hash"].nunique() == 1
    assert (registry["target_end_idx"] - registry["target_start_idx"]).eq(2).all()
    assert (registry["history_end_idx"] - registry["history_start_idx"]).eq(3).all()
    assert (registry["target_start_idx"] == registry["start_idx"]).all()


def test_write_registry_outputs_writes_csv_config_and_manifest(tmp_path: Path) -> None:
    codebook = pd.DataFrame(
        {
            "official_cluster_code": [0, 24],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_highF"],
        }
    )
    cfg = RegistryConfig(
        history_len=3,
        pred_len=2,
        sample_stride=2,
        split_context_policy="strict_within_split",
        channels=("ind_1",),
    )
    registry, manifest = build_window_registry({"hour": _toy_df()}, codebook, cfg)

    out_dir = write_registry_outputs(registry, manifest, cfg, tmp_path)

    assert (out_dir / "window_index.csv").exists()
    assert (out_dir / "config.yml").exists()
    assert (out_dir / "manifest.json").exists()

    loaded = pd.read_csv(out_dir / "window_index.csv")
    assert len(loaded) == len(registry)


def test_smoke_registry_outputs_do_not_overwrite_full_registry(tmp_path: Path) -> None:
    codebook = pd.DataFrame(
        {
            "official_cluster_code": [0, 24],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_highF"],
        }
    )
    cfg = RegistryConfig(history_len=3, pred_len=2, sample_stride=2, channels=("ind_1",))
    registry, manifest = build_window_registry({"hour": _toy_df()}, codebook, cfg)

    full_dir = write_registry_outputs(registry, manifest, cfg, tmp_path)
    smoke_dir = write_registry_outputs(
        registry,
        manifest,
        cfg,
        tmp_path,
        run_scope="smoke",
        max_items_per_subset=1,
    )

    assert full_dir != smoke_dir
    assert smoke_dir.name.endswith("__smoke_max_items_1")
    smoke_manifest = pd.read_json(smoke_dir / "manifest.json", typ="series")
    assert smoke_manifest["run_scope"] == "smoke"
    assert smoke_manifest["max_items_per_subset"] == 1


def test_load_official_codebook_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "codebook.csv"
    pd.DataFrame(
        {
            "official_cluster_code": [24],
            "official_tsf_cell": ["lowT_lowS_highF"],
        }
    ).to_csv(path, index=False)

    loaded = load_official_codebook(path)

    assert loaded.loc[24, "official_tsf_cell"] == "lowT_lowS_highF"


def test_iter_window_offsets_matches_quito_overlap_boundaries() -> None:
    cfg = RegistryConfig(history_len=3, pred_len=2, sample_stride=1, split_context_policy="quito_overlap")
    bounds = {"train": (0, 8), "valid": (8, 10), "test": (10, 14)}

    assert iter_window_offsets("train", *bounds["train"], cfg.history_len, cfg.pred_len, cfg.sample_stride, cfg.split_context_policy) == [3, 4, 5, 6]
    assert iter_window_offsets("valid", *bounds["valid"], cfg.history_len, cfg.pred_len, cfg.sample_stride, cfg.split_context_policy) == [8]
    assert iter_window_offsets("test", *bounds["test"], cfg.history_len, cfg.pred_len, cfg.sample_stride, cfg.split_context_policy) == [10, 11, 12]


def test_quito_overlap_registry_keeps_target_inside_split_and_allows_context_overlap() -> None:
    codebook = pd.DataFrame(
        {
            "official_cluster_code": [0, 24],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_highF"],
        }
    )
    cfg = RegistryConfig(history_len=3, pred_len=2, sample_stride=1, channels=("ind_1",))

    registry, manifest = build_window_registry({"hour": _toy_df()}, codebook, cfg)

    assert manifest["split_window_counts"] == {"train": 8, "valid": 2, "test": 6}
    valid_rows = registry[registry["split"] == "valid"]
    test_rows = registry[registry["split"] == "test"]
    assert valid_rows["target_start_idx"].eq(8).all()
    assert valid_rows["history_start_idx"].eq(5).all()
    assert (valid_rows["history_start_idx"] < valid_rows["split_start_idx"]).all()
    assert set(test_rows["target_start_idx"]) == {10, 11, 12}
    assert (registry["target_start_idx"] >= registry["split_start_idx"]).all()
    assert (registry["target_end_idx"] <= registry["split_end_idx"]).all()


def test_physical_window_id_is_independent_of_sample_stride() -> None:
    codebook = pd.DataFrame(
        {
            "official_cluster_code": [0, 24],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_highF"],
        }
    )
    cfg_dense = RegistryConfig(history_len=3, pred_len=2, sample_stride=1, channels=("ind_1",))
    cfg_sparse = RegistryConfig(history_len=3, pred_len=2, sample_stride=2, channels=("ind_1",))

    dense, _ = build_window_registry({"hour": _toy_df()}, codebook, cfg_dense)
    sparse, _ = build_window_registry({"hour": _toy_df()}, codebook, cfg_sparse)
    join_cols = ["subset", "item_id", "channel", "split", "target_start_idx", "history_len", "pred_len"]
    merged = dense.merge(sparse, on=join_cols, suffixes=("_dense", "_sparse"))

    assert not merged.empty
    assert (merged["physical_window_id_dense"] == merged["physical_window_id_sparse"]).all()
    assert (merged["sample_set_id_dense"] != merged["sample_set_id_sparse"]).all()
