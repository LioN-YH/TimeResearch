from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from tools.quitobench_proxy_imageization_latency import (
    LatencySweepConfig,
    benchmark_online_components,
    build_latency_manifest,
    write_latency_outputs,
)


def test_benchmark_online_components_reports_required_latency_columns() -> None:
    histories = torch.arange(4 * 192, dtype=torch.float32).reshape(4, 192)
    periods = [24, 24, 144, 144]
    config = LatencySweepConfig(warmup_iters=0, measure_iters=1)

    row = benchmark_online_components(
        histories=histories,
        periods=periods,
        device="cpu",
        batch_size=4,
        config=config,
    )

    assert row["device"] == "cpu"
    assert row["batch_size"] == 4
    assert row["sampled_windows"] == 4
    assert row["proxy_output_shape"] == [4, 15]
    assert row["view_tensor_shape"] == [4, 3, 64, 192]
    assert row["proxy_torch_latency_ms_per_window"] >= 0.0
    assert row["view_tensor_latency_ms_per_window"] >= 0.0
    assert row["proxy_plus_view_latency_ms_per_window"] >= 0.0
    assert row["runs_expert_models"] is False
    assert row["implements_router"] is False
    assert row["recomputes_stage1_1_cache"] is False


def test_write_latency_outputs_writes_csv_and_manifest(tmp_path: Path) -> None:
    rows = [
        {
            "device": "cpu",
            "batch_size": 1,
            "sampled_windows": 1,
            "proxy_torch_latency_ms_per_window": 0.1,
            "view_tensor_latency_ms_per_window": 0.2,
            "proxy_plus_view_latency_ms_per_window": 0.3,
            "runs_expert_models": False,
            "implements_router": False,
            "recomputes_stage1_1_cache": False,
        }
    ]
    manifest = build_latency_manifest(
        rows=rows,
        config=LatencySweepConfig(warmup_iters=0, measure_iters=1),
        sample_set_id="sample_a",
        base_registry_id="base_a",
        input_registry_dir=Path("registry"),
        sampled_windows=1,
        cuda_available=False,
    )

    csv_path, manifest_path = write_latency_outputs(rows=rows, manifest=manifest, output_dir=tmp_path)

    assert csv_path.name == "stage1_2b_proxy_imageization_latency.csv"
    assert manifest_path.name == "stage1_2b_proxy_imageization_latency_manifest.json"
    loaded = pd.read_csv(csv_path)
    assert loaded["proxy_torch_latency_ms_per_window"].iloc[0] == 0.1
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["stage"] == "stage1_2b_proxy_imageization_latency_sweep"
    assert loaded_manifest["runs_expert_models"] is False
    assert loaded_manifest["implements_router"] is False
    assert loaded_manifest["recomputes_stage1_1_cache"] is False
