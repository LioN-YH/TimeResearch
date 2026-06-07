# Stage 1.4a Lightweight Expert Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 1.4a minimal expert prediction cache for four history-only lightweight experts, with prediction/error/oracle outputs keyed by `physical_window_id`.

**Architecture:** Add one focused CLI module under `tools/` that loads the Stage 1.0 registry, extracts history/target from QuitoBench data, computes lightweight expert predictions, evaluates target-only errors, writes cache files and manifest, and emits profiling summaries. Add one focused pytest module that locks schema, history-only behavior, expert formulas, uniqueness, manifest flags, and output writing.

**Tech Stack:** Python, pandas, numpy, pyarrow/parquet via pandas, pytest, existing QuitoBench helpers from `tools.quitobench_window_registry`.

---

## File Structure

Create:

- `tools/quitobench_lightweight_expert_cache.py`
  - Owns Stage 1.4a lightweight expert prediction, error computation, oracle summary, cache writing, and CLI.
  - Must not import or run visual encoder, router, or neural expert frameworks.
  - Must use Chinese docstrings/comments for project-specific explanations.

- `tests/test_quitobench_lightweight_expert_cache.py`
  - Unit tests for toy histories/targets, schema validation, output writing, and manifest flags.

- `experiment_logs/YYYY-MM-DD_HHMM_stage1_4a_lightweight_expert_cache.md`
  - Created during execution run, not during initial implementation.
  - Must be registered in `experiment_logs/实验日志总览.md`.

Modify during execution:

- `experiment_logs/实验日志总览.md`
  - Add one row after the Stage 1.4a smoke/full execution log exists.

Do not modify:

- Quito upstream code under `quito/`.
- Stage 1.0/1.1/1.2 output files.
- Router/gate code, because Stage 1.4a must not implement router.

Default output root:

```text
outputs/vision_ts_routing/expert_predictions/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/
    lightweight_v1/
      predictions.parquet
      errors.parquet
      manifest.json
      profiling/
        cell_model_matrix.csv
        oracle_summary.csv
```

---

### Task 1: Define Tests for Lightweight Expert Formulas and Error Metrics

**Files:**

- Create: `tests/test_quitobench_lightweight_expert_cache.py`
- Create later in Task 2: `tools/quitobench_lightweight_expert_cache.py`

- [ ] **Step 1: Write the failing test file with imports and toy data**

Create `tests/test_quitobench_lightweight_expert_cache.py` with this initial content:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.quitobench_lightweight_expert_cache import (
    EXPERT_IDS,
    REQUIRED_REGISTRY_COLUMNS,
    build_cache_manifest,
    compute_error_table,
    compute_lightweight_expert_predictions,
    compute_oracle_summary,
    validate_registry,
    write_expert_cache_outputs,
)


def _toy_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "physical_window_id": "w_1",
                "window_id": "w_1",
                "base_registry_id": "base_v1",
                "sample_set_id": "sample_v1",
                "subset": "hour",
                "split": "valid",
                "item_id": "item_a",
                "channel": "ind_1",
                "period": 4,
                "official_tsf_cell": "lowT_highS_highF",
                "history_start_idx": 0,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 12,
                "history_len": 8,
                "pred_len": 4,
            },
            {
                "physical_window_id": "w_2",
                "window_id": "w_2",
                "base_registry_id": "base_v1",
                "sample_set_id": "sample_v1",
                "subset": "hour",
                "split": "valid",
                "item_id": "item_b",
                "channel": "ind_1",
                "period": 4,
                "official_tsf_cell": "highT_lowS_highF",
                "history_start_idx": 0,
                "history_end_idx": 8,
                "target_start_idx": 8,
                "target_end_idx": 12,
                "history_len": 8,
                "pred_len": 4,
            },
        ]
    )
```

- [ ] **Step 2: Add formula expectations**

Append these tests:

```python
def test_required_registry_columns_include_stage1_keys() -> None:
    assert "physical_window_id" in REQUIRED_REGISTRY_COLUMNS
    assert "sample_set_id" in REQUIRED_REGISTRY_COLUMNS
    assert "target_start_idx" in REQUIRED_REGISTRY_COLUMNS
    assert "pred_len" in REQUIRED_REGISTRY_COLUMNS


def test_compute_lightweight_expert_predictions_uses_history_only() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)

    assert set(predictions["expert_id"]) == set(EXPERT_IDS)
    assert len(predictions) == 4
    assert predictions[["physical_window_id", "expert_id"]].duplicated().sum() == 0

    wide_cols = ["yhat_0", "yhat_1", "yhat_2", "yhat_3"]
    by_expert = predictions.set_index("expert_id")

    assert by_expert.loc["last_value", wide_cols].to_numpy(dtype=float).tolist() == [40.0, 40.0, 40.0, 40.0]
    assert by_expert.loc["seasonal_naive", wide_cols].to_numpy(dtype=float).tolist() == [10.0, 20.0, 30.0, 40.0]
    assert by_expert.loc["recent_mean", wide_cols].to_numpy(dtype=float).tolist() == [25.0, 25.0, 25.0, 25.0]

    linear = by_expert.loc["linear_trend", wide_cols].to_numpy(dtype=float)
    np.testing.assert_allclose(linear, np.array([42.0, 47.0, 52.0, 57.0]), atol=1e-8)
```

The `linear_trend` expectation is based on fitting `y = -3 + 5x` to the eight history points after least-squares smoothing on the toy sequence.

- [ ] **Step 3: Add error and oracle tests**

Append:

```python
def test_compute_error_table_and_oracle_summary() -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    targets = {"w_1": np.array([10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)

    errors = compute_error_table(predictions, targets)

    assert set(errors["expert_id"]) == set(EXPERT_IDS)
    seasonal = errors.set_index("expert_id").loc["seasonal_naive"]
    assert seasonal["mse"] == pytest.approx(0.0)
    assert seasonal["mae"] == pytest.approx(0.0)
    assert bool(seasonal["is_oracle_top1"]) is True

    weights = errors.groupby("physical_window_id")["soft_oracle_weight"].sum()
    assert weights.loc["w_1"] == pytest.approx(1.0)

    summary = compute_oracle_summary(errors)
    assert summary.loc[0, "num_windows"] == 1
    assert summary.loc[0, "oracle_mse"] == pytest.approx(0.0)
    assert summary.loc[0, "best_fixed_expert"] == "seasonal_naive"
```

- [ ] **Step 4: Run the test to verify it fails because the module does not exist**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'tools.quitobench_lightweight_expert_cache'
```

---

### Task 2: Implement Core Lightweight Expert Functions

**Files:**

- Create: `tools/quitobench_lightweight_expert_cache.py`
- Test: `tests/test_quitobench_lightweight_expert_cache.py`

- [ ] **Step 1: Create the module header, constants, and registry validation**

Create `tools/quitobench_lightweight_expert_cache.py`:

```python
"""Stage 1.4a：QuitoBench sample-channel 轻量专家预测缓存。

本脚本只运行 history-only 的极轻量专家，生成专家预测、误差和 oracle
profiling 缓存。不训练视觉 encoder，不实现 router，不运行 neural experts。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_window_registry import DEFAULT_DATA_DIR, load_subset_frames


DEFAULT_SAMPLE_SET_ID = "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e"
DEFAULT_REGISTRY_DIR = ROOT / "outputs/vision_ts_routing/window_registry" / DEFAULT_SAMPLE_SET_ID
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/expert_predictions" / DEFAULT_SAMPLE_SET_ID

EXPERT_IDS = ("last_value", "seasonal_naive", "recent_mean", "linear_trend")
EXPERT_FAMILY = {
    "last_value": "statistical_baseline",
    "seasonal_naive": "statistical_baseline",
    "recent_mean": "statistical_baseline",
    "linear_trend": "statistical_baseline",
}

REQUIRED_REGISTRY_COLUMNS = {
    "physical_window_id",
    "window_id",
    "base_registry_id",
    "sample_set_id",
    "subset",
    "split",
    "item_id",
    "channel",
    "period",
    "official_tsf_cell",
    "history_start_idx",
    "history_end_idx",
    "target_start_idx",
    "target_end_idx",
    "history_len",
    "pred_len",
}


@dataclass(frozen=True)
class LightweightExpertConfig:
    """Stage 1.4a 轻量专家缓存配置。"""

    stage: str = "stage1_4a_lightweight_expert_cache"
    expert_set_id: str = "lightweight_v1"
    recent_mean_fraction: float = 0.25
    soft_oracle_temperature: float = 1.0
    eps: float = 1e-8
    random_seed: int = 20260607


def validate_registry(registry: pd.DataFrame) -> None:
    missing = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少必要列：{sorted(missing)}")
    if not registry["physical_window_id"].is_unique:
        raise ValueError("registry 中 physical_window_id 不唯一")
    if (registry["pred_len"].astype(int) <= 0).any():
        raise ValueError("registry 中 pred_len 必须为正整数")
    if (registry["history_len"].astype(int) <= 0).any():
        raise ValueError("registry 中 history_len 必须为正整数")
```

- [ ] **Step 2: Add expert formula helpers**

Append:

```python
def _finite_history(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    filled = arr.astype(float, copy=True)
    mean_value = float(np.mean(filled[finite]))
    filled[~finite] = mean_value
    return filled


def _last_value(history: np.ndarray, pred_len: int) -> np.ndarray:
    return np.full(pred_len, float(history[-1]), dtype=float)


def _seasonal_naive(history: np.ndarray, pred_len: int, period: int) -> np.ndarray:
    period = int(period)
    if period <= 0:
        return _last_value(history, pred_len)
    base = history[-period:] if len(history) >= period else history
    repeats = int(math.ceil(pred_len / len(base)))
    return np.tile(base, repeats)[:pred_len].astype(float)


def _recent_mean(history: np.ndarray, pred_len: int, fraction: float) -> np.ndarray:
    width = max(1, int(math.ceil(len(history) * fraction)))
    value = float(np.mean(history[-width:]))
    return np.full(pred_len, value, dtype=float)


def _linear_trend(history: np.ndarray, pred_len: int) -> np.ndarray:
    if len(history) < 2:
        return _last_value(history, pred_len)
    x = np.arange(len(history), dtype=float)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return _last_value(history, pred_len)
    slope = float(np.dot(x_centered, history - float(np.mean(history))) / denom)
    intercept = float(np.mean(history) - slope * np.mean(x))
    future_x = np.arange(len(history), len(history) + pred_len, dtype=float)
    return intercept + slope * future_x
```

- [ ] **Step 3: Add prediction table builder**

Append:

```python
def compute_lightweight_expert_predictions(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    config: LightweightExpertConfig | None = None,
) -> pd.DataFrame:
    """对每个 physical_window_id 只用 history 计算四个轻量专家预测。"""

    validate_registry(registry)
    cfg = config or LightweightExpertConfig()
    rows: list[dict[str, object]] = []

    for row in registry.itertuples(index=False):
        physical_window_id = str(row.physical_window_id)
        if physical_window_id not in histories:
            raise KeyError(f"缺少 history：{physical_window_id}")
        history = _finite_history(histories[physical_window_id])
        pred_len = int(row.pred_len)
        if len(history) != int(row.history_len):
            raise ValueError(f"{physical_window_id} history 长度 {len(history)} != {int(row.history_len)}")

        expert_predictions = {
            "last_value": _last_value(history, pred_len),
            "seasonal_naive": _seasonal_naive(history, pred_len, int(row.period)),
            "recent_mean": _recent_mean(history, pred_len, cfg.recent_mean_fraction),
            "linear_trend": _linear_trend(history, pred_len),
        }

        for expert_id, prediction in expert_predictions.items():
            out_row = {
                "physical_window_id": physical_window_id,
                "window_id": str(row.window_id),
                "base_registry_id": str(row.base_registry_id),
                "sample_set_id": str(row.sample_set_id),
                "subset": str(row.subset),
                "split": str(row.split),
                "item_id": str(row.item_id),
                "channel": str(row.channel),
                "period": int(row.period),
                "official_tsf_cell": str(row.official_tsf_cell),
                "history_start_idx": int(row.history_start_idx),
                "target_start_idx": int(row.target_start_idx),
                "pred_len": pred_len,
                "expert_id": expert_id,
                "expert_family": EXPERT_FAMILY[expert_id],
                "prediction_format": "wide_columns",
            }
            for horizon_idx, value in enumerate(prediction):
                out_row[f"yhat_{horizon_idx}"] = float(value)
            rows.append(out_row)

    predictions = pd.DataFrame(rows)
    if predictions[["physical_window_id", "expert_id"]].duplicated().any():
        raise ValueError("predictions 中 (physical_window_id, expert_id) 不唯一")
    return predictions
```

- [ ] **Step 4: Run the formula tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py::test_compute_lightweight_expert_predictions_uses_history_only -q
```

Expected:

```text
1 passed
```

---

### Task 3: Implement Error Table and Oracle Summary

**Files:**

- Modify: `tools/quitobench_lightweight_expert_cache.py`
- Test: `tests/test_quitobench_lightweight_expert_cache.py`

- [ ] **Step 1: Add prediction column helper and error computation**

Append:

```python
def _prediction_columns(predictions: pd.DataFrame) -> list[str]:
    cols = [col for col in predictions.columns if col.startswith("yhat_")]
    return sorted(cols, key=lambda name: int(name.split("_", 1)[1]))


def compute_error_table(
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
    config: LightweightExpertConfig | None = None,
) -> pd.DataFrame:
    """用 target 计算误差和 soft oracle；target 不参与专家输入。"""

    cfg = config or LightweightExpertConfig()
    yhat_cols = _prediction_columns(predictions)
    if not yhat_cols:
        raise ValueError("predictions 缺少 yhat_* 预测列")

    rows: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        row_dict = row._asdict()
        physical_window_id = str(row_dict["physical_window_id"])
        if physical_window_id not in targets:
            raise KeyError(f"缺少 target：{physical_window_id}")
        target = np.asarray(targets[physical_window_id], dtype=float)
        yhat = np.asarray([row_dict[col] for col in yhat_cols], dtype=float)
        if len(target) != len(yhat):
            raise ValueError(f"{physical_window_id} target 长度 {len(target)} != prediction 长度 {len(yhat)}")
        diff = yhat - target
        rows.append(
            {
                "physical_window_id": physical_window_id,
                "sample_set_id": str(row_dict["sample_set_id"]),
                "split": str(row_dict["split"]),
                "subset": str(row_dict["subset"]),
                "item_id": str(row_dict["item_id"]),
                "channel": str(row_dict["channel"]),
                "official_tsf_cell": str(row_dict["official_tsf_cell"]),
                "expert_id": str(row_dict["expert_id"]),
                "mse": float(np.mean(diff * diff)),
                "mae": float(np.mean(np.abs(diff))),
            }
        )

    errors = pd.DataFrame(rows)
    errors["rank_in_window"] = errors.groupby("physical_window_id")["mse"].rank(method="first", ascending=True).astype(int)
    errors["is_oracle_top1"] = errors["rank_in_window"] == 1

    def _softmax(group: pd.DataFrame) -> pd.Series:
        values = -group["mse"].to_numpy(dtype=float) / max(float(cfg.soft_oracle_temperature), cfg.eps)
        values = values - float(np.max(values))
        weights = np.exp(values)
        weights = weights / max(float(weights.sum()), cfg.eps)
        return pd.Series(weights, index=group.index)

    errors["soft_oracle_weight"] = errors.groupby("physical_window_id", group_keys=False).apply(_softmax)
    return errors
```

- [ ] **Step 2: Add oracle summary**

Append:

```python
def compute_oracle_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """汇总 oracle ensemble、best fixed expert 和 uniform ensemble 的窗口级上界。"""

    if errors.empty:
        return pd.DataFrame(
            [
                {
                    "num_windows": 0,
                    "num_experts": 0,
                    "oracle_mse": np.nan,
                    "best_fixed_expert": "",
                    "best_fixed_mse": np.nan,
                    "uniform_ensemble_mse_proxy": np.nan,
                    "oracle_gap_vs_best_fixed": np.nan,
                }
            ]
        )

    oracle_mse = float(errors.groupby("physical_window_id")["mse"].min().mean())
    fixed = errors.groupby("expert_id")["mse"].mean().sort_values()
    best_fixed_expert = str(fixed.index[0])
    best_fixed_mse = float(fixed.iloc[0])
    uniform_proxy = float(errors.groupby("physical_window_id")["mse"].mean().mean())
    return pd.DataFrame(
        [
            {
                "num_windows": int(errors["physical_window_id"].nunique()),
                "num_experts": int(errors["expert_id"].nunique()),
                "oracle_mse": oracle_mse,
                "best_fixed_expert": best_fixed_expert,
                "best_fixed_mse": best_fixed_mse,
                "uniform_ensemble_mse_proxy": uniform_proxy,
                "oracle_gap_vs_best_fixed": best_fixed_mse - oracle_mse,
            }
        ]
    )
```

- [ ] **Step 3: Run the error/oracle test**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py::test_compute_error_table_and_oracle_summary -q
```

Expected:

```text
1 passed
```

---

### Task 4: Add Output Writing, Manifest, and Profiling Tests

**Files:**

- Modify: `tests/test_quitobench_lightweight_expert_cache.py`
- Modify: `tools/quitobench_lightweight_expert_cache.py`

- [ ] **Step 1: Add tests for registry validation and output writing**

Append to `tests/test_quitobench_lightweight_expert_cache.py`:

```python
def test_validate_registry_rejects_duplicate_physical_window_id() -> None:
    registry = pd.concat([_toy_registry().iloc[[0]], _toy_registry().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="physical_window_id 不唯一"):
        validate_registry(registry)


def test_write_expert_cache_outputs_writes_expected_files(tmp_path: Path) -> None:
    registry = _toy_registry().iloc[[0]].reset_index(drop=True)
    histories = {"w_1": np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)}
    targets = {"w_1": np.array([10, 20, 30, 40], dtype=float)}
    predictions = compute_lightweight_expert_predictions(registry, histories)
    errors = compute_error_table(predictions, targets)
    oracle = compute_oracle_summary(errors)
    cell_matrix = errors.groupby(["official_tsf_cell", "expert_id"], as_index=False)["mse"].mean()
    manifest = build_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=0.5,
        input_registry_dir=Path("/tmp/registry"),
        max_rows=1,
    )

    out_dir = write_expert_cache_outputs(
        predictions=predictions,
        errors=errors,
        oracle_summary=oracle,
        cell_model_matrix=cell_matrix,
        manifest=manifest,
        output_root=tmp_path,
        expert_set_id="lightweight_v1",
    )

    assert (out_dir / "predictions.parquet").exists()
    assert (out_dir / "errors.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "profiling/cell_model_matrix.csv").exists()
    assert (out_dir / "profiling/oracle_summary.csv").exists()

    loaded_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["stage"] == "stage1_4a_lightweight_expert_cache"
    assert loaded_manifest["expert_set_id"] == "lightweight_v1"
    assert loaded_manifest["implements_router"] is False
    assert loaded_manifest["runs_visual_encoder"] is False
    assert loaded_manifest["runs_neural_experts"] is False
    assert loaded_manifest["future_read_policy"] == "history_only_for_prediction"

    loaded_predictions = pd.read_parquet(out_dir / "predictions.parquet")
    assert loaded_predictions[["physical_window_id", "expert_id"]].duplicated().sum() == 0
```

- [ ] **Step 2: Add manifest and output functions**

Append to `tools/quitobench_lightweight_expert_cache.py`:

```python
def build_cell_model_matrix(errors: pd.DataFrame) -> pd.DataFrame:
    """按 official TSF cell 汇总每个 expert 的平均误差和胜率。"""

    grouped = (
        errors.groupby(["official_tsf_cell", "expert_id"], as_index=False)
        .agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            oracle_top1_rate=("is_oracle_top1", "mean"),
            num_windows=("physical_window_id", "nunique"),
        )
        .sort_values(["official_tsf_cell", "mse", "expert_id"])
        .reset_index(drop=True)
    )
    grouped["rank_in_cell"] = grouped.groupby("official_tsf_cell")["mse"].rank(method="first", ascending=True).astype(int)
    return grouped


def build_cache_manifest(
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    elapsed_seconds: float,
    input_registry_dir: Path,
    max_rows: int | None,
    config: LightweightExpertConfig | None = None,
) -> dict[str, object]:
    cfg = config or LightweightExpertConfig()
    return {
        "stage": cfg.stage,
        "expert_set_id": cfg.expert_set_id,
        "expert_ids": list(EXPERT_IDS),
        "expert_families": EXPERT_FAMILY,
        "sample_set_id": sorted(registry["sample_set_id"].astype(str).unique().tolist()),
        "base_registry_id": sorted(registry["base_registry_id"].astype(str).unique().tolist()),
        "input_registry_dir": str(input_registry_dir),
        "max_rows": max_rows,
        "total_windows": int(registry["physical_window_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "error_rows": int(len(errors)),
        "unique_prediction_key": bool(not predictions[["physical_window_id", "expert_id"]].duplicated().any()),
        "unique_error_key": bool(not errors[["physical_window_id", "expert_id"]].duplicated().any()),
        "split_window_counts": registry.groupby("split")["physical_window_id"].nunique().to_dict(),
        "subset_window_counts": registry.groupby("subset")["physical_window_id"].nunique().to_dict(),
        "cell_window_counts": registry.groupby("official_tsf_cell")["physical_window_id"].nunique().to_dict(),
        "prediction_format": "wide_columns",
        "future_read_policy": "history_only_for_prediction",
        "target_usage": "error_and_oracle_only",
        "implements_router": False,
        "runs_visual_encoder": False,
        "runs_neural_experts": False,
        "modifies_quito_code": False,
        "elapsed_seconds": float(elapsed_seconds),
        "latency_ms_per_window": float(elapsed_seconds * 1000.0 / max(len(registry), 1)),
        "config": asdict(cfg),
        "output_files": {
            "predictions": "predictions.parquet",
            "errors": "errors.parquet",
            "manifest": "manifest.json",
            "cell_model_matrix": "profiling/cell_model_matrix.csv",
            "oracle_summary": "profiling/oracle_summary.csv",
        },
    }


def write_expert_cache_outputs(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    cell_model_matrix: pd.DataFrame,
    manifest: Mapping[str, object],
    output_root: Path,
    expert_set_id: str,
) -> Path:
    out_dir = output_root / expert_set_id
    profiling_dir = out_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    errors.to_parquet(out_dir / "errors.parquet", index=False)
    cell_model_matrix.to_csv(profiling_dir / "cell_model_matrix.csv", index=False)
    oracle_summary.to_csv(profiling_dir / "oracle_summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir
```

- [ ] **Step 3: Run the output tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py::test_write_expert_cache_outputs_writes_expected_files -q
```

Expected:

```text
1 passed
```

---

### Task 5: Add Registry/Data Loading and CLI

**Files:**

- Modify: `tools/quitobench_lightweight_expert_cache.py`
- Test: `tests/test_quitobench_lightweight_expert_cache.py`

- [ ] **Step 1: Add loading helpers**

Append to `tools/quitobench_lightweight_expert_cache.py`:

```python
def load_registry(registry_dir: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    registry_path = registry_dir / "window_index.csv"
    manifest_path = registry_dir / "manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"registry 不存在：{registry_path}")
    registry = pd.read_csv(registry_path)
    if max_rows is not None:
        registry = registry.head(max_rows).copy()
    validate_registry(registry)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return registry, manifest


def extract_histories_and_targets(
    registry: pd.DataFrame,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按 registry 边界从 QuitoBench 数据中抽取 history 和 target。"""

    frames = load_subset_frames(data_dir)
    histories: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for row in registry.itertuples(index=False):
        subset_frame = frames[str(row.subset)]
        item_frame = subset_frame[subset_frame["item_id"] == row.item_id].sort_values("date")
        values = item_frame[str(row.channel)].to_numpy(dtype=float)
        history = values[int(row.history_start_idx) : int(row.history_end_idx)]
        target = values[int(row.target_start_idx) : int(row.target_end_idx)]
        if len(history) != int(row.history_len):
            raise ValueError(f"{row.physical_window_id} history 长度 {len(history)} != {int(row.history_len)}")
        if len(target) != int(row.pred_len):
            raise ValueError(f"{row.physical_window_id} target 长度 {len(target)} != {int(row.pred_len)}")
        histories[str(row.physical_window_id)] = history
        targets[str(row.physical_window_id)] = target
    return histories, targets
```

- [ ] **Step 2: Add CLI parse and main**

Append:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expert-set-id", default="lightweight_v1")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--recent-mean-fraction", type=float, default=0.25)
    parser.add_argument("--soft-oracle-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LightweightExpertConfig(
        expert_set_id=args.expert_set_id,
        recent_mean_fraction=args.recent_mean_fraction,
        soft_oracle_temperature=args.soft_oracle_temperature,
    )
    start = time.perf_counter()
    registry, registry_manifest = load_registry(args.registry_dir, max_rows=args.max_rows)
    histories, targets = extract_histories_and_targets(registry, data_dir=args.data_dir)
    predictions = compute_lightweight_expert_predictions(registry, histories, config=config)
    errors = compute_error_table(predictions, targets, config=config)
    oracle_summary = compute_oracle_summary(errors)
    cell_model_matrix = build_cell_model_matrix(errors)
    elapsed = time.perf_counter() - start
    manifest = build_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=elapsed,
        input_registry_dir=args.registry_dir,
        max_rows=args.max_rows,
        config=config,
    )
    manifest["input_registry_manifest"] = registry_manifest
    out_dir = write_expert_cache_outputs(
        predictions=predictions,
        errors=errors,
        oracle_summary=oracle_summary,
        cell_model_matrix=cell_model_matrix,
        manifest=manifest,
        output_root=args.output_root,
        expert_set_id=args.expert_set_id,
    )
    print(f"[done] output_dir={out_dir}")
    print(f"[done] windows={manifest['total_windows']}")
    print(f"[done] prediction_rows={manifest['prediction_rows']}")
    print(f"[done] latency_ms_per_window={manifest['latency_ms_per_window']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all Stage 1.4a unit tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

Expected:

```text
5 passed
```

---

### Task 6: Run Smoke Cache and Record Experiment Log

**Files:**

- Runtime output: `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512/`
- Create: `experiment_logs/YYYY-MM-DD_HHMM_stage1_4a_lightweight_expert_cache.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: Run smoke cache**

Run:

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --max-rows 512 \
  --expert-set-id lightweight_v1__smoke_max_rows_512
```

Expected terminal shape:

```text
[done] output_dir=...
[done] windows=512
[done] prediction_rows=2048
[done] latency_ms_per_window=...
```

- [ ] **Step 2: Inspect smoke manifest and uniqueness**

Run:

```bash
conda run -n quito python - <<'PY'
import json
from pathlib import Path
import pandas as pd

out = Path("outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512")
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
pred = pd.read_parquet(out / "predictions.parquet")
err = pd.read_parquet(out / "errors.parquet")
print("windows", manifest["total_windows"])
print("prediction_rows", len(pred))
print("error_rows", len(err))
print("prediction_unique", pred[["physical_window_id", "expert_id"]].duplicated().sum() == 0)
print("error_unique", err[["physical_window_id", "expert_id"]].duplicated().sum() == 0)
print("soft_weight_max_abs_error", float((err.groupby("physical_window_id")["soft_oracle_weight"].sum() - 1.0).abs().max()))
print("implements_router", manifest["implements_router"])
print("runs_neural_experts", manifest["runs_neural_experts"])
PY
```

Expected:

```text
windows 512
prediction_rows 2048
error_rows 2048
prediction_unique True
error_unique True
soft_weight_max_abs_error 0.0
implements_router False
runs_neural_experts False
```

Small floating-point output such as `2.220446049250313e-16` is acceptable for `soft_weight_max_abs_error`.

- [ ] **Step 3: Write experiment log**

Create `experiment_logs/YYYY-MM-DD_HHMM_stage1_4a_lightweight_expert_cache.md` with:

```markdown
# Stage 1.4a：轻量专家预测缓存 smoke

## 1. 目的

验证 Stage 1.4a 专家预测缓存 schema、history-only 轻量专家、误差计算、soft oracle 和 cell-level profiling，不实现 router，不运行视觉 encoder，不运行 neural experts。

## 2. 输入

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- max_rows: `512`
- expert_set_id: `lightweight_v1__smoke_max_rows_512`

## 3. 命令

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --max-rows 512 \
  --expert-set-id lightweight_v1__smoke_max_rows_512
```

## 4. 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512/
```

包含：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

## 5. 验证结果

记录实际运行输出：

- windows:
- prediction_rows:
- error_rows:
- prediction_unique:
- error_unique:
- soft_weight_max_abs_error:
- implements_router:
- runs_neural_experts:

## 6. 结论

Stage 1.4a smoke 若通过，说明专家预测缓存主键、输出 schema 和 oracle error 计算可以工作。后续可扩大到正式 working registry，或先接入 Stage 1.3a visual encoder adapter smoke。
```

Replace the blank values with actual command output.

- [ ] **Step 4: Update experiment log overview**

Append one row to `experiment_logs/实验日志总览.md` using the existing table/list style. The row must mention:

```text
Stage 1.4a lightweight expert cache smoke；不实现 router；不运行 neural experts；输出 lightweight_v1__smoke_max_rows_512。
```

---

### Task 7: Run Broader Verification and Commit

**Files:**

- Modify staged files from Tasks 1-6.

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 2: Run relevant regression tests**

Run:

```bash
conda run -n quito python -m pytest \
  tests/test_quitobench_lightweight_expert_cache.py \
  tests/test_quitobench_window_registry.py \
  tests/test_quitobench_sample_channel_light_proxy.py \
  -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Check git diff**

Run:

```bash
git diff -- tools/quitobench_lightweight_expert_cache.py tests/test_quitobench_lightweight_expert_cache.py experiment_logs/实验日志总览.md experiment_logs/*_stage1_4a_lightweight_expert_cache.md
```

Verify:

- No router/gate implementation appears.
- No visual encoder training appears.
- No neural expert framework is invoked.
- `physical_window_id` and `sample_set_id` are present in prediction and error paths.
- Comments and project-specific docs are in Chinese.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add \
  tools/quitobench_lightweight_expert_cache.py \
  tests/test_quitobench_lightweight_expert_cache.py \
  experiment_logs/实验日志总览.md \
  experiment_logs/*_stage1_4a_lightweight_expert_cache.md
git commit -m "feat: add stage 1.4a lightweight expert cache"
```

Expected:

```text
[main <hash>] feat: add stage 1.4a lightweight expert cache
```

---

## Self-Review

Spec coverage:

- Framework reuse boundary: covered by using only project protocol code in 1.4a and leaving neural framework reuse to 1.4b.
- `physical_window_id` primary key: covered by tests and prediction/error uniqueness checks.
- `sample_set_id` retention: covered by prediction/error schema and manifest.
- History-only prediction: covered by formulas that accept only history and by manifest `future_read_policy`.
- Target use only for error/oracle: covered by `compute_error_table`.
- No router / no visual encoder / no neural experts: covered by manifest flags and diff checklist.
- Profiling output: covered by `cell_model_matrix.csv` and `oracle_summary.csv`.
- Experiment logging: covered by Task 6.

Placeholder scan target:

```bash
python - <<'PY'
from pathlib import Path

path = Path("docs/superpowers/plans/2026-06-07-stage1-4a-lightweight-expert-cache.md")
needles = [
    "T" + "BD",
    "TO" + "DO",
    "FIX" + "ME",
    "待" + "定",
    "占" + "位",
    "implement " + "later",
    "fill " + "in",
]
text = path.read_text(encoding="utf-8")
for needle in needles:
    if needle in text:
        print(f"FOUND {needle}")
PY
```

Expected: no output.
