from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.quitobench_window_registry import (
    RegistryConfig,
    build_config_hash,
    build_window_registry,
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


def test_config_hash_is_stable_and_order_independent() -> None:
    cfg_a = RegistryConfig(history_len=3, pred_len=2, stride=2, channels=("ind_1", "ind_2"))
    cfg_b = RegistryConfig(channels=("ind_1", "ind_2"), stride=2, pred_len=2, history_len=3)

    assert build_config_hash(cfg_a) == build_config_hash(cfg_b)
    assert len(build_config_hash(cfg_a)) == 12


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
    cfg = RegistryConfig(history_len=3, pred_len=2, stride=2, channels=("ind_1", "ind_2"))

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
    cfg = RegistryConfig(history_len=3, pred_len=2, stride=2, channels=("ind_1",))
    registry, manifest = build_window_registry({"hour": _toy_df()}, codebook, cfg)

    out_dir = write_registry_outputs(registry, manifest, cfg, tmp_path)

    assert (out_dir / "window_index.csv").exists()
    assert (out_dir / "config.yml").exists()
    assert (out_dir / "manifest.json").exists()

    loaded = pd.read_csv(out_dir / "window_index.csv")
    assert len(loaded) == len(registry)


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
