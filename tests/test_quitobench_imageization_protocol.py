from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.quitobench_imageization_protocol import (
    ImageizationConfig,
    imageize_batch,
    normalize_history_batch,
    sample_smoke_registry,
    write_imageization_outputs,
)


def _toy_registry() -> pd.DataFrame:
    rows = []
    for subset in ["hour", "min"]:
        period = 24 if subset == "hour" else 144
        for split in ["train", "valid"]:
            for cell in ["highT_highS_highF", "lowT_lowS_lowF"]:
                for i in range(3):
                    rows.append(
                        {
                            "physical_window_id": f"{subset}_{split}_{cell}_{i}",
                            "window_id": f"{subset}_{split}_{cell}_{i}",
                            "base_registry_id": "base_a",
                            "sample_set_id": "sample_a",
                            "subset": subset,
                            "split": split,
                            "item_id": 100 + i,
                            "channel": "ind_1",
                            "period": period,
                            "official_tsf_cell": cell,
                            "history_start_idx": 0,
                            "history_end_idx": 192,
                            "target_start_idx": 192,
                            "target_end_idx": 288,
                            "history_len": 192,
                            "pred_len": 96,
                        }
                    )
    return pd.DataFrame(rows)


def test_normalize_history_batch_is_per_window() -> None:
    config = ImageizationConfig(height=8, width=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [100.0, 100.0, 100.0, 100.0]])

    normalized, meta = normalize_history_batch(x, config)

    assert normalized.shape == x.shape
    assert meta["mean"].shape == (2,)
    assert meta["std"].shape == (2,)
    assert meta["mean"].tolist() == [2.5, 100.0]
    assert torch.isfinite(normalized).all()
    assert normalized[1].abs().sum().item() == 0.0


def test_imageize_batch_outputs_three_view_tensor_and_period_padding() -> None:
    config = ImageizationConfig()
    x = torch.arange(384, dtype=torch.float32).reshape(2, 192)

    tensor, meta = imageize_batch(x, periods=[24, 144], config=config)

    assert tensor.shape == (2, 3, 64, 192)
    assert meta["view_names"] == ["line_raster", "period_fold", "fft_power"]
    assert meta["view_tensor_semantics"] == "multi_view_not_rgb"
    assert meta["padding_lengths"] == [0, 96]
    assert meta["num_cycles"] == [8, 2]
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_sample_smoke_registry_limits_each_subset_split_cell_group() -> None:
    sampled = sample_smoke_registry(_toy_registry(), max_per_group=2, random_seed=7)

    counts = sampled.groupby(["subset", "split", "official_tsf_cell"]).size()
    assert counts.max() == 2
    assert len(sampled) == 16
    assert sampled["physical_window_id"].is_unique


def test_write_imageization_outputs_writes_tensor_index_manifest_and_debug_png(tmp_path: Path) -> None:
    registry = sample_smoke_registry(_toy_registry(), max_per_group=1, random_seed=7)
    tensor = torch.zeros((len(registry), 3, 8, 12), dtype=torch.float32)
    image_index = registry[
        [
            "physical_window_id",
            "sample_set_id",
            "subset",
            "split",
            "item_id",
            "channel",
            "period",
        ]
    ].copy()
    image_index["mean"] = 0.0
    image_index["std"] = 1.0
    manifest = {
        "stage": "stage1_2_imageization_protocol_smoke",
        "sample_set_id": "sample_a",
        "view_names": ["line_raster", "period_fold", "fft_power"],
        "view_dim": 3,
        "tensor_shape": [len(registry), 3, 8, 12],
        "view_tensor_semantics": "multi_view_not_rgb",
        "normalization": {"scope": "per_physical_window_id_history"},
    }

    out_dir = write_imageization_outputs(
        view_tensor=tensor,
        image_index=image_index,
        manifest=manifest,
        output_root=tmp_path,
        debug_png_count=2,
    )

    assert out_dir.name == "sample_a__stage1_2_smoke_v1"
    loaded_npz = np.load(out_dir / "view_tensor_sample.npz")
    assert loaded_npz["view_tensor"].shape == (len(registry), 3, 8, 12)
    loaded_index = pd.read_csv(out_dir / "image_index.csv")
    assert loaded_index["physical_window_id"].is_unique
    loaded_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["view_tensor_semantics"] == "multi_view_not_rgb"
    assert len(list((out_dir / "debug_png").glob("*.png"))) == 2
