from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.quitobench_sample_channel_light_proxy import (
    FEATURE_COLUMNS,
    ProxyConfig,
    compute_light_proxy_features,
    compute_light_proxy_torch,
    compute_window_proxy,
    write_proxy_outputs,
)


def _toy_subset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_time": pd.date_range("2023-01-01", periods=8, freq="h"),
            "item_id": [101] * 8,
            "cluster": [0] * 8,
            "ind_1": [1.0, 2.0, 3.0, 4.0, 1000.0, 1001.0, 1002.0, 1003.0],
        }
    )


def _toy_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physical_window_id": ["w_train_1"],
            "window_id": ["w_train_1"],
            "base_registry_id": ["base_a"],
            "sample_set_id": ["sample_a"],
            "subset": ["hour"],
            "split": ["train"],
            "item_id": [101],
            "channel": ["ind_1"],
            "period": [24],
            "history_start_idx": [0],
            "history_end_idx": [4],
            "target_start_idx": [4],
            "target_end_idx": [6],
            "history_len": [4],
            "pred_len": [2],
        }
    )


def test_compute_window_proxy_uses_only_history_values() -> None:
    proxy = compute_window_proxy([1.0, 2.0, 3.0, 4.0], period=24)

    assert proxy["mean"] == 2.5
    assert proxy["last_value"] == 4.0
    assert proxy["max"] == 4.0
    assert proxy["missing_ratio"] == 0.0
    assert proxy["amplitude"] == 3.0
    assert proxy["slope"] > 0.0


def test_compute_light_proxy_features_keeps_physical_window_id_and_sample_set_id() -> None:
    proxy, manifest = compute_light_proxy_features(
        registry=_toy_registry(),
        subset_frames={"hour": _toy_subset_frame()},
        config=ProxyConfig(),
    )

    assert proxy["physical_window_id"].tolist() == ["w_train_1"]
    assert proxy["sample_set_id"].tolist() == ["sample_a"]
    assert proxy["mean"].iloc[0] == 2.5
    assert proxy["max"].iloc[0] == 4.0
    assert proxy["target_start_idx"].iloc[0] == 4
    assert manifest["total_windows"] == 1
    assert "spectral_entropy" in manifest["feature_columns"]


def test_compute_light_proxy_features_rejects_duplicate_physical_window_id() -> None:
    duplicated = pd.concat([_toy_registry(), _toy_registry()], ignore_index=True)

    try:
        compute_light_proxy_features(
            registry=duplicated,
            subset_frames={"hour": _toy_subset_frame()},
            config=ProxyConfig(),
        )
    except ValueError as exc:
        assert "physical_window_id 不唯一" in str(exc)
    else:
        raise AssertionError("重复 physical_window_id 应触发错误")


def test_write_proxy_outputs_writes_feature_table_and_manifest(tmp_path: Path) -> None:
    proxy, manifest = compute_light_proxy_features(
        registry=_toy_registry(),
        subset_frames={"hour": _toy_subset_frame()},
        config=ProxyConfig(),
    )

    out_dir = write_proxy_outputs(proxy, manifest, tmp_path, output_format="csv")

    assert out_dir.name == "sample_a"
    assert (out_dir / "sample_channel_proxy.csv").exists()
    assert (out_dir / "manifest.json").exists()
    loaded = pd.read_csv(out_dir / "sample_channel_proxy.csv")
    assert loaded["physical_window_id"].tolist() == ["w_train_1"]


def test_smoke_proxy_outputs_do_not_overwrite_full_outputs(tmp_path: Path) -> None:
    proxy, manifest = compute_light_proxy_features(
        registry=_toy_registry(),
        subset_frames={"hour": _toy_subset_frame()},
        config=ProxyConfig(),
    )

    full_dir = write_proxy_outputs(proxy, manifest, tmp_path, output_format="csv")
    smoke_dir = write_proxy_outputs(
        proxy,
        manifest,
        tmp_path,
        output_format="csv",
        run_scope="smoke",
        max_rows=1,
    )

    assert full_dir != smoke_dir
    assert smoke_dir.name.endswith("__smoke_max_rows_1")
    smoke_manifest = pd.read_json(smoke_dir / "manifest.json", typ="series")
    assert smoke_manifest["run_scope"] == "smoke"
    assert smoke_manifest["max_rows"] == 1


def test_compute_light_proxy_torch_matches_numpy_reference_feature_order() -> None:
    histories = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, float("nan"), 1.0, 2.0, 3.0, 5.0],
        ],
        dtype=torch.float32,
    )
    periods = torch.tensor([2, 3], dtype=torch.int64)

    torch_proxy = compute_light_proxy_torch(histories, periods)
    expected = np.array(
        [
            [compute_window_proxy(row.tolist(), period=int(period))[feature] for feature in FEATURE_COLUMNS]
            for row, period in zip(histories, periods, strict=True)
        ],
        dtype=np.float32,
    )

    assert torch_proxy.shape == (2, len(FEATURE_COLUMNS))
    assert torch_proxy.device.type == "cpu"
    np.testing.assert_allclose(torch_proxy.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


def test_compute_light_proxy_torch_runs_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    histories = torch.tensor(
        [
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    periods = torch.tensor([2, 4], dtype=torch.int64)

    cpu_proxy = compute_light_proxy_torch(histories, periods)
    cuda_proxy = compute_light_proxy_torch(histories.cuda(), periods.cuda())

    assert cuda_proxy.device.type == "cuda"
    torch.testing.assert_close(cuda_proxy.cpu(), cpu_proxy, rtol=1e-5, atol=1e-5)
