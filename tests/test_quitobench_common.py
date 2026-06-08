from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.quitobench_common import (
    ensure_unique_key,
    filter_common_expert_windows,
    load_json_manifest,
    prediction_columns,
    require_columns,
    write_json_manifest,
)


def test_prediction_columns_sorts_by_horizon_index() -> None:
    frame = pd.DataFrame(columns=["yhat_10", "physical_window_id", "yhat_2", "yhat_0"])

    assert prediction_columns(frame) == ["yhat_0", "yhat_2", "yhat_10"]


def test_filter_common_expert_windows_keeps_only_complete_windows() -> None:
    frame = pd.DataFrame(
        [
            {"physical_window_id": "w1", "expert_id": "a", "value": 1},
            {"physical_window_id": "w1", "expert_id": "b", "value": 2},
            {"physical_window_id": "w2", "expert_id": "a", "value": 3},
            {"physical_window_id": "w3", "expert_id": "a", "value": 4},
            {"physical_window_id": "w3", "expert_id": "b", "value": 5},
            {"physical_window_id": "w3", "expert_id": "extra", "value": 6},
        ]
    )

    filtered = filter_common_expert_windows(frame, required_experts=("a", "b"))

    assert filtered["physical_window_id"].tolist() == ["w1", "w1", "w3", "w3"]
    assert set(filtered["expert_id"]) == {"a", "b"}


def test_json_manifest_helpers_round_trip_and_missing_file_defaults(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {"sample_set_id": "sample", "value": 3}

    write_json_manifest(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert load_json_manifest(path) == payload
    assert load_json_manifest(tmp_path / "missing.json") == {}


def test_require_columns_reports_missing_columns() -> None:
    frame = pd.DataFrame(columns=["physical_window_id"])

    try:
        require_columns(frame, {"physical_window_id", "expert_id"}, label="predictions")
    except ValueError as exc:
        assert "predictions 缺少列" in str(exc)
        assert "expert_id" in str(exc)
    else:
        raise AssertionError("require_columns should fail for missing columns")


def test_ensure_unique_key_rejects_duplicate_keys() -> None:
    frame = pd.DataFrame(
        [
            {"physical_window_id": "w1", "expert_id": "a"},
            {"physical_window_id": "w1", "expert_id": "a"},
        ]
    )

    try:
        ensure_unique_key(frame, ["physical_window_id", "expert_id"], label="errors")
    except ValueError as exc:
        assert "errors 存在重复键" in str(exc)
        assert "w1" in str(exc)
    else:
        raise AssertionError("ensure_unique_key should fail for duplicate keys")
