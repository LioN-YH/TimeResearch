from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.quitobench_registry_audit import build_registry_audit, write_registry_audit_outputs


def _toy_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "physical_window_id": "w1",
                "sample_set_id": "sample_a",
                "base_registry_id": "base_a",
                "split": "train",
                "subset": "hour",
                "official_tsf_cell": "cell_a",
                "item_id": 1,
                "channel": "ind_1",
            },
            {
                "physical_window_id": "w2",
                "sample_set_id": "sample_a",
                "base_registry_id": "base_a",
                "split": "valid",
                "subset": "hour",
                "official_tsf_cell": "cell_a",
                "item_id": 1,
                "channel": "ind_2",
            },
            {
                "physical_window_id": "w3",
                "sample_set_id": "sample_a",
                "base_registry_id": "base_a",
                "split": "test",
                "subset": "min",
                "official_tsf_cell": "cell_b",
                "item_id": 2,
                "channel": "ind_1",
            },
        ]
    )


def test_build_registry_audit_summarizes_core_distribution() -> None:
    summary, tables = build_registry_audit(_toy_registry(), manifest={"total_windows": 3})

    assert summary["total_rows"] == 3
    assert summary["unique_physical_windows"] == 3
    assert summary["unique_items"] == 2
    assert summary["unique_channels"] == 2
    assert summary["sample_set_ids"] == ["sample_a"]
    assert summary["manifest_total_windows"] == 3
    assert summary["min_split_subset_cell_windows"] == 1
    assert set(tables) == {
        "split_counts",
        "subset_counts",
        "cell_counts",
        "split_subset_cell_counts",
        "item_channel_counts",
    }
    assert tables["split_subset_cell_counts"]["num_windows"].sum() == 3


def test_write_registry_audit_outputs_writes_summary_and_tables(tmp_path: Path) -> None:
    summary, tables = build_registry_audit(_toy_registry())

    out_dir = write_registry_audit_outputs(summary, tables, tmp_path)

    assert out_dir == tmp_path
    assert json.loads((tmp_path / "audit_summary.json").read_text(encoding="utf-8"))["total_rows"] == 3
    assert (tmp_path / "split_subset_cell_counts.csv").exists()
    assert (tmp_path / "item_channel_counts.csv").exists()
