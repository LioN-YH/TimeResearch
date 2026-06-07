from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.quitobench_visual_encoder_adapter_smoke import (
    TinyViewCnnEncoder,
    VisualEncoderSmokeConfig,
    build_embedding_table,
    encode_view_tensor,
    load_stage12_view_tensor,
    write_visual_embedding_outputs,
)


def test_tiny_view_cnn_encoder_outputs_embedding_shape() -> None:
    config = VisualEncoderSmokeConfig(embedding_dim=32)
    encoder = TinyViewCnnEncoder(config)
    view_tensor = torch.linspace(0.0, 1.0, steps=4 * 3 * 64 * 192, dtype=torch.float32).reshape(4, 3, 64, 192)

    embeddings = encoder(view_tensor)

    assert embeddings.shape == (4, 32)
    assert torch.isfinite(embeddings).all()


def test_encode_view_tensor_is_deterministic_for_same_weights() -> None:
    config = VisualEncoderSmokeConfig(embedding_dim=16, random_seed=123)
    view_tensor = torch.ones((3, 3, 64, 192), dtype=torch.float32)

    first, first_meta = encode_view_tensor(view_tensor, config=config, device="cpu")
    second, second_meta = encode_view_tensor(view_tensor, config=config, device="cpu")

    assert first.shape == (3, 16)
    assert torch.allclose(first, second)
    assert first_meta["encoder_id"] == "tiny_view_cnn_v1"
    assert second_meta["embedding_dim"] == 16


def _toy_image_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physical_window_id": ["w0", "w1", "w2"],
            "window_id": ["w0", "w1", "w2"],
            "base_registry_id": ["base_a", "base_a", "base_a"],
            "sample_set_id": ["sample_a", "sample_a", "sample_a"],
            "subset": ["hour", "hour", "min"],
            "split": ["train", "valid", "test"],
            "item_id": [1, 2, 3],
            "channel": ["ind_1", "ind_2", "ind_3"],
            "period": [24, 24, 144],
            "official_tsf_cell": ["highT_highS_highF", "highT_highS_lowF", "lowT_lowS_lowF"],
            "view_tensor_row": [0, 1, 2],
        }
    )


def test_build_embedding_table_keeps_physical_window_id_and_wide_columns() -> None:
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    table = build_embedding_table(_toy_image_index(), embeddings, encoder_id="tiny_view_cnn_v1")

    assert table["physical_window_id"].tolist() == ["w0", "w1", "w2"]
    assert table["sample_set_id"].nunique() == 1
    assert table["encoder_id"].unique().tolist() == ["tiny_view_cnn_v1"]
    assert table[["z_0", "z_1", "z_2", "z_3"]].shape == (3, 4)
    assert table["physical_window_id"].is_unique


def test_write_visual_embedding_outputs_writes_expected_files(tmp_path: Path) -> None:
    embeddings = torch.ones((3, 4), dtype=torch.float32)
    table = build_embedding_table(_toy_image_index(), embeddings, encoder_id="tiny_view_cnn_v1")
    manifest = {
        "stage": "stage1_3a0_visual_embedding_cache_smoke",
        "sample_set_id": "sample_a",
        "encoder_id": "tiny_view_cnn_v1",
        "embedding_dim": 4,
        "num_windows": 3,
        "trains_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
    }
    latency_rows = [{"device": "cpu", "batch_size": 3, "encoder_latency_ms_per_window": 0.1}]

    out_dir = write_visual_embedding_outputs(
        embedding_table=table,
        image_index=_toy_image_index(),
        latency_rows=latency_rows,
        manifest=manifest,
        output_root=tmp_path,
    )

    assert out_dir.parent.name == "sample_a"
    assert out_dir.name == "visual_embedding_cache_smoke_v1"
    assert (out_dir / "embeddings.parquet").exists()
    assert (out_dir / "embedding_index.csv").exists()
    assert (out_dir / "latency.csv").exists()
    assert (out_dir / "manifest.json").exists()
    loaded_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["implements_router"] is False


def test_load_stage12_view_tensor_validates_index_and_tensor(tmp_path: Path) -> None:
    image_dir = tmp_path / "stage12"
    image_dir.mkdir()
    np.savez_compressed(image_dir / "view_tensor_sample.npz", view_tensor=np.zeros((2, 3, 64, 192), dtype=np.float32))
    pd.DataFrame(
        {
            "physical_window_id": ["w0", "w1"],
            "window_id": ["w0", "w1"],
            "base_registry_id": ["base_a", "base_a"],
            "sample_set_id": ["sample_a", "sample_a"],
            "subset": ["hour", "min"],
            "split": ["train", "test"],
            "item_id": [1, 2],
            "channel": ["ind_1", "ind_2"],
            "period": [24, 144],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_lowF"],
            "view_tensor_row": [0, 1],
        }
    ).to_csv(image_dir / "image_index.csv", index=False)
    (image_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "stage1_2_imageization_protocol_smoke",
                "sample_set_id": "sample_a",
                "image_protocol_id": "view3_h64_w192_v1",
                "view_tensor_semantics": "multi_view_not_rgb",
                "normalization": {"scope": "per_physical_window_id_history"},
                "tensor_shape": [2, 3, 64, 192],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    view_tensor, image_index, manifest = load_stage12_view_tensor(image_dir)

    assert view_tensor.shape == (2, 3, 64, 192)
    assert image_index["physical_window_id"].tolist() == ["w0", "w1"]
    assert manifest["view_tensor_semantics"] == "multi_view_not_rgb"
