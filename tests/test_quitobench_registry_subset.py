from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.quitobench_registry_subset import materialize_registry_subset


def _toy_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [
        ("train", "hour", "cell_a"),
        ("train", "min", "cell_b"),
        ("valid", "hour", "cell_a"),
        ("test", "min", "cell_b"),
    ]
    for group_idx, (split, subset, cell) in enumerate(groups):
        for offset in range(3):
            idx = group_idx * 3 + offset
            rows.append(
                {
                    "dataset": "hq-bench/quitobench",
                    "revision": "rev",
                    "data_version": "v",
                    "subset": subset,
                    "item_id": 100 + group_idx,
                    "channel": f"ind_{offset + 1}",
                    "split": split,
                    "target_start_idx": idx * 10 + 96,
                    "history_len": 96,
                    "pred_len": 48,
                    "start_idx": idx * 10 + 96,
                    "sample_stride": 288,
                    "split_context_policy": "quito_overlap",
                    "base_registry_id": "base_registry_a",
                    "sample_set_id": "base_sample_a",
                    "config_hash": "abc",
                    "physical_window_id": f"physical_{idx:03d}",
                    "window_id": f"physical_{idx:03d}",
                    "period": 24,
                    "official_cluster": group_idx,
                    "official_tsf_cell": cell,
                    "history_start_idx": idx * 10,
                    "history_end_idx": idx * 10 + 96,
                    "target_end_idx": idx * 10 + 144,
                    "history_start_time": "2024-01-01T00:00:00",
                    "history_end_time": "2024-01-04T23:00:00",
                    "target_start_time": "2024-01-05T00:00:00",
                    "target_end_time": "2024-01-06T23:00:00",
                    "split_start_idx": 0,
                    "split_end_idx": 1000,
                    "item_length": 1000,
                    "split_length": 1000,
                }
            )
    return pd.DataFrame(rows)


def _write_registry_dir(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True)
    frame.to_csv(path / "window_index.csv", index=False)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "sample_set_id": "base_sample_a",
                "base_registry_id": "base_registry_a",
                "total_windows": len(frame),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "config.yml").write_text("history_len: 96\npred_len: 48\n", encoding="utf-8")


def test_materialize_registry_subset_writes_fixed_stratified_registry(tmp_path: Path) -> None:
    input_dir = tmp_path / "input_registry"
    output_dir = tmp_path / "output_registry"
    _write_registry_dir(input_dir, _toy_registry())

    manifest = materialize_registry_subset(
        input_registry_dir=input_dir,
        output_registry_dir=output_dir,
        sample_set_id="sample_matrix_v1",
        target_rows=8,
        stratify_cols=["split", "subset", "official_tsf_cell"],
        random_seed=20260608,
    )

    out = pd.read_csv(output_dir / "window_index.csv")
    assert len(out) == 8
    assert out["physical_window_id"].is_unique
    assert out["sample_set_id"].unique().tolist() == ["sample_matrix_v1"]
    assert out["base_registry_id"].unique().tolist() == ["base_registry_a"]
    assert set(out["physical_window_id"]).issubset(set(_toy_registry()["physical_window_id"]))
    assert (
        out.groupby(["split", "subset", "official_tsf_cell"])["physical_window_id"].nunique().tolist()
        == [2, 2, 2, 2]
    )
    assert (output_dir / "config.yml").read_text(encoding="utf-8") == "history_len: 96\npred_len: 48\n"
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert saved_manifest["stage"] == "canonical_expert_matrix_registry_subset"
    assert saved_manifest["sample_set_id"] == "sample_matrix_v1"
    assert saved_manifest["base_sample_set_id"] == "base_sample_a"
    assert saved_manifest["target_rows"] == 8
    assert saved_manifest["selected_rows"] == 8
    assert saved_manifest["stratify_cols"] == ["split", "subset", "official_tsf_cell"]
    assert saved_manifest["random_seed"] == 20260608
    assert saved_manifest["split_window_counts"] == {"test": 2, "train": 4, "valid": 2}
    assert saved_manifest["subset_window_counts"] == {"hour": 4, "min": 4}
    assert saved_manifest["cell_window_counts"] == {"cell_a": 4, "cell_b": 4}
    assert saved_manifest["unique_items"] == 4
    assert saved_manifest["unique_channels"] == ["ind_1", "ind_2", "ind_3"]


def test_materialize_registry_subset_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    input_dir = tmp_path / "input_registry"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_registry_dir(input_dir, _toy_registry())

    materialize_registry_subset(
        input_registry_dir=input_dir,
        output_registry_dir=first_dir,
        sample_set_id="sample_matrix_v1",
        target_rows=8,
        stratify_cols=["split", "subset", "official_tsf_cell"],
        random_seed=7,
    )
    materialize_registry_subset(
        input_registry_dir=input_dir,
        output_registry_dir=second_dir,
        sample_set_id="sample_matrix_v1",
        target_rows=8,
        stratify_cols=["split", "subset", "official_tsf_cell"],
        random_seed=7,
    )

    first = pd.read_csv(first_dir / "window_index.csv")
    second = pd.read_csv(second_dir / "window_index.csv")
    assert first["physical_window_id"].tolist() == second["physical_window_id"].tolist()


def test_materialize_registry_subset_rejects_duplicate_physical_window_ids(tmp_path: Path) -> None:
    input_dir = tmp_path / "input_registry"
    frame = _toy_registry()
    frame.loc[1, "physical_window_id"] = frame.loc[0, "physical_window_id"]
    _write_registry_dir(input_dir, frame)

    with pytest.raises(ValueError, match="physical_window_id"):
        materialize_registry_subset(
            input_registry_dir=input_dir,
            output_registry_dir=tmp_path / "out",
            sample_set_id="sample_matrix_v1",
            target_rows=8,
            stratify_cols=["split", "subset", "official_tsf_cell"],
            random_seed=1,
        )


def test_materialize_registry_subset_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "input_registry"
    output_dir = tmp_path / "output_registry"
    _write_registry_dir(input_dir, _toy_registry())
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        materialize_registry_subset(
            input_registry_dir=input_dir,
            output_registry_dir=output_dir,
            sample_set_id="sample_matrix_v1",
            target_rows=8,
            stratify_cols=["split", "subset", "official_tsf_cell"],
            random_seed=1,
        )


def test_cli_help_runs_when_invoked_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "tools/quitobench_registry_subset.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--input-registry-dir" in result.stdout
