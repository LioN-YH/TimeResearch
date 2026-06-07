# Stage 1.4e PatchTST Official Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the runner diagnostics and training controls needed to verify whether PatchTST's weak Stage 1.4c/1.4d result comes from implementation/scale mismatch or from the `192/96/S` task itself.

**Architecture:** Keep Quito model code unchanged. Extend the project wrapper with optional train-set standardization, scheduler/drop-last controls, prediction diagnostics, and manifest metadata; then run a small official-aligned DLinear/PatchTST sanity matrix.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, Quito local package, parquet cache outputs.

---

## File Structure

- Modify `tools/quitobench_framework_expert_cache.py`: add scaler dataclass/functions, dataset scaling support, DataLoader and scheduler flags, CLI args, and manifest metadata.
- Modify `tests/test_quitobench_dlinear_expert_cache.py`: add tests for scaler computation, inverse transform, drop-last/scheduler args, and cache-scale prediction.
- Create `tools/quitobench_expert_prediction_diagnostics.py`: summarize prediction/target/error scale for existing expert cache outputs.
- Create `tests/test_quitobench_expert_prediction_diagnostics.py`: test diagnostics on toy parquet data.
- Create `experiment_logs/2026-06-07_HHMM_stage1_4e_patchtst_official_alignment.md`: record implementation changes, commands, results, and interpretation.
- Modify `experiment_logs/实验日志总览.md`: register the Stage 1.4e log.

---

### Task 1: Add Train-Set Scaling Tests

**Files:**
- Modify: `tests/test_quitobench_dlinear_expert_cache.py`
- Modify: `tools/quitobench_framework_expert_cache.py`

- [ ] **Step 1: Write failing tests for scaler behavior**

Add imports in `tests/test_quitobench_dlinear_expert_cache.py`:

```python
from tools.quitobench_framework_expert_cache import (
    WindowStandardizer,
    build_train_split_standardizer,
    apply_standardizer_to_series_maps,
)
```

Add tests:

```python
def test_build_train_split_standardizer_uses_only_train_windows() -> None:
    registry = _toy_registry()
    histories, targets = _toy_histories_targets()

    standardizer = build_train_split_standardizer(registry, histories, targets)

    train_values = np.concatenate(
        [
            histories["w_1"],
            targets["w_1"],
            histories["w_2"],
            targets["w_2"],
        ]
    )
    expected_mean = float(np.mean(train_values))
    expected_std = float(np.std(train_values) + 1e-8)
    assert standardizer.mean == pytest.approx(expected_mean)
    assert standardizer.std == pytest.approx(expected_std)
    assert standardizer.scope == "train_split_global_window_values"


def test_apply_standardizer_round_trips_histories_and_targets() -> None:
    histories, targets = _toy_histories_targets()
    standardizer = WindowStandardizer(mean=5.0, std=2.0, scope="test")

    scaled_histories, scaled_targets = apply_standardizer_to_series_maps(histories, targets, standardizer)

    assert scaled_histories["w_1"][0] == pytest.approx(-2.0)
    restored = standardizer.inverse_transform(scaled_targets["w_1"])
    np.testing.assert_allclose(restored, targets["w_1"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py::test_build_train_split_standardizer_uses_only_train_windows tests/test_quitobench_dlinear_expert_cache.py::test_apply_standardizer_round_trips_histories_and_targets -q
```

Expected: FAIL because `WindowStandardizer` and helper functions do not exist.

- [ ] **Step 3: Implement minimal scaler helpers**

Add to `tools/quitobench_framework_expert_cache.py` near the config dataclasses:

```python
@dataclass(frozen=True)
class WindowStandardizer:
    """Stage 1.4e wrapper-level train split standardizer."""

    mean: float
    std: float
    scope: str

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - np.float32(self.mean)) / np.float32(self.std)

    def inverse_transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * np.float32(self.std) + np.float32(self.mean)
```

Add helper functions:

```python
def build_train_split_standardizer(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
) -> WindowStandardizer:
    train_ids = registry.loc[registry["split"].astype(str) == "train", "physical_window_id"].astype(str).tolist()
    if not train_ids:
        raise ValueError("train-set standardizer 需要至少一个 train window")
    values = [np.asarray(histories[physical_window_id], dtype=np.float32) for physical_window_id in train_ids]
    values.extend(np.asarray(targets[physical_window_id], dtype=np.float32) for physical_window_id in train_ids)
    merged = np.concatenate(values)
    return WindowStandardizer(
        mean=float(np.mean(merged)),
        std=float(np.std(merged) + 1e-8),
        scope="train_split_global_window_values",
    )


def apply_standardizer_to_series_maps(
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    standardizer: WindowStandardizer | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if standardizer is None:
        return (
            {key: np.asarray(value, dtype=np.float32) for key, value in histories.items()},
            {key: np.asarray(value, dtype=np.float32) for key, value in targets.items()},
        )
    return (
        {key: standardizer.transform(value) for key, value in histories.items()},
        {key: standardizer.transform(value) for key, value in targets.items()},
    )
```

- [ ] **Step 4: Run scaler tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py::test_build_train_split_standardizer_uses_only_train_windows tests/test_quitobench_dlinear_expert_cache.py::test_apply_standardizer_round_trips_histories_and_targets -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/quitobench_framework_expert_cache.py tests/test_quitobench_dlinear_expert_cache.py
git commit -m "feat: add expert runner train split scaler"
```

---

### Task 2: Wire Scaler Into Training, Prediction, and Manifest

**Files:**
- Modify: `tools/quitobench_framework_expert_cache.py`
- Modify: `tests/test_quitobench_dlinear_expert_cache.py`

- [ ] **Step 1: Write failing CLI and prediction-scale tests**

Add test:

```python
def test_parse_args_exposes_stage14e_alignment_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--train-set-standardize",
            "--drop-last",
            "--scheduler",
            "cosine",
            "--eta-min",
            "0.00001",
            "--num-workers",
            "2",
            "--eval-batch-size",
            "64",
        ],
    )

    args = parse_args()

    assert args.train_set_standardize is True
    assert args.drop_last is True
    assert args.scheduler == "cosine"
    assert args.eta_min == pytest.approx(0.00001)
    assert args.num_workers == 2
    assert args.eval_batch_size == 64
```

Add test:

```python
def test_predict_with_model_inverse_transforms_standardized_predictions() -> None:
    class EchoLastModel:
        def eval(self) -> None:
            return None

        def predict(self, x, y=None):
            return x[:, -4:, :]

    registry = _toy_registry().iloc[[0]].copy()
    histories, targets = _toy_histories_targets()
    standardizer = WindowStandardizer(mean=10.0, std=2.0, scope="test")
    scaled_histories, scaled_targets = apply_standardizer_to_series_maps(histories, targets, standardizer)

    predictions = predict_with_model(
        EchoLastModel(),
        registry,
        scaled_histories,
        scaled_targets,
        config=DLinearExpertConfig(seq_len=8, pred_len=4, batch_size=1),
        device="cpu",
        output_standardizer=standardizer,
    )

    np.testing.assert_allclose(predictions["w_1"], histories["w_1"][-4:])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py::test_parse_args_exposes_stage14e_alignment_flags tests/test_quitobench_dlinear_expert_cache.py::test_predict_with_model_inverse_transforms_standardized_predictions -q
```

Expected: FAIL because CLI args and `output_standardizer` are not implemented.

- [ ] **Step 3: Add config fields and train loop flags**

Add fields to each expert config dataclass:

```python
train_set_standardize: bool = False
drop_last: bool = False
scheduler: str = "none"
eta_min: float = 1e-5
num_workers: int = 0
eval_batch_size: int | None = None
```

Update `_train_model()` signature with `drop_last`, `scheduler`, `eta_min`, and `num_workers`. DataLoader should use:

```python
loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=drop_last,
    num_workers=num_workers,
)
```

After optimizer creation:

```python
lr_scheduler = None
if scheduler == "cosine":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=eta_min,
    )
elif scheduler != "none":
    raise ValueError(f"不支持的 scheduler：{scheduler}")
```

After each epoch:

```python
if lr_scheduler is not None:
    lr_scheduler.step()
```

Training stats should include:

```python
"drop_last": bool(drop_last),
"scheduler": scheduler,
"eta_min": float(eta_min),
"num_workers": int(num_workers),
"final_learning_rate": float(optimizer.param_groups[0]["lr"]),
```

- [ ] **Step 4: Add prediction inverse transform**

Change `predict_with_model()` signature:

```python
def predict_with_model(
    model: DLinear,
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    config: DLinearExpertConfig | PatchTSTExpertConfig | TSMixerExpertConfig | None = None,
    device: str = "cpu",
    output_standardizer: WindowStandardizer | None = None,
) -> dict[str, np.ndarray]:
```

Use eval batch size:

```python
batch_size = cfg.eval_batch_size or cfg.batch_size
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
```

Before storing prediction:

```python
values = pred[:, 0].astype(np.float32)
if output_standardizer is not None:
    values = output_standardizer.inverse_transform(values)
predictions[physical_window_id] = values.astype(float)
```

- [ ] **Step 5: Wire CLI and main flow**

Add parser args:

```python
parser.add_argument("--train-set-standardize", action="store_true", default=False)
parser.add_argument("--drop-last", action="store_true", default=False)
parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
parser.add_argument("--eta-min", type=float, default=1e-5)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-batch-size", type=int, default=None)
```

Pass those values into all three config constructors.

In `main()` after `extract_histories_and_targets()`:

```python
standardizer = build_train_split_standardizer(registry, histories, targets) if config.train_set_standardize else None
model_histories, model_targets = apply_standardizer_to_series_maps(histories, targets, standardizer)
```

Train with `model_histories/model_targets`, predict with `model_histories/model_targets`, compute errors against original `targets`.

Manifest should include:

```python
manifest["standardization"] = (
    asdict(standardizer)
    if standardizer is not None
    else {"enabled": False}
)
```

- [ ] **Step 6: Run targeted and full runner tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/quitobench_framework_expert_cache.py tests/test_quitobench_dlinear_expert_cache.py
git commit -m "feat: align expert runner training controls"
```

---

### Task 3: Add Expert Prediction Diagnostics Tool

**Files:**
- Create: `tools/quitobench_expert_prediction_diagnostics.py`
- Create: `tests/test_quitobench_expert_prediction_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics tests**

Create `tests/test_quitobench_expert_prediction_diagnostics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.quitobench_expert_prediction_diagnostics import summarize_prediction_scale


def test_summarize_prediction_scale_reports_prediction_target_and_error_stats() -> None:
    predictions = pd.DataFrame(
        {
            "physical_window_id": ["w1", "w2"],
            "expert_id": ["patchtst_quito", "patchtst_quito"],
            "yhat_0": [1.0, 10.0],
            "yhat_1": [2.0, 20.0],
        }
    )
    targets = {
        "w1": np.array([1.5, 2.5], dtype=float),
        "w2": np.array([9.0, 19.0], dtype=float),
    }

    summary = summarize_prediction_scale(predictions, targets)

    assert summary["rows"] == 2
    assert summary["horizon_columns"] == 2
    assert summary["prediction"]["max"] == pytest.approx(20.0)
    assert summary["target"]["min"] == pytest.approx(1.5)
    assert summary["absolute_error"]["max"] == pytest.approx(1.0)
    assert summary["finite_prediction_rate"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_expert_prediction_diagnostics.py -q
```

Expected: FAIL because the tool does not exist.

- [ ] **Step 3: Implement diagnostics tool**

Create `tools/quitobench_expert_prediction_diagnostics.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tools.quitobench_lightweight_expert_cache import extract_histories_and_targets, load_registry


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan, "p999": np.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "p999": float(np.quantile(finite, 0.999)),
    }


def summarize_prediction_scale(
    predictions: pd.DataFrame,
    targets: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    horizon_cols = sorted(
        [col for col in predictions.columns if col.startswith("yhat_")],
        key=lambda value: int(value.split("_", 1)[1]),
    )
    pred_values = predictions[horizon_cols].to_numpy(dtype=float)
    target_values = np.stack(
        [np.asarray(targets[str(row.physical_window_id)], dtype=float) for row in predictions.itertuples(index=False)]
    )
    abs_error = np.abs(pred_values - target_values)
    return {
        "rows": int(len(predictions)),
        "horizon_columns": int(len(horizon_cols)),
        "expert_ids": sorted(predictions["expert_id"].astype(str).unique().tolist()) if "expert_id" in predictions else [],
        "finite_prediction_rate": float(np.isfinite(pred_values).mean()),
        "prediction": _stats(pred_values),
        "target": _stats(target_values),
        "absolute_error": _stats(abs_error),
    }
```

Also add a CLI that accepts `--cache-dir`, `--registry-dir`, `--data-dir`, and `--output-json`, loads `predictions.parquet`, extracts targets, writes JSON, and prints the output path.

- [ ] **Step 4: Run diagnostics tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_expert_prediction_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/quitobench_expert_prediction_diagnostics.py tests/test_quitobench_expert_prediction_diagnostics.py
git commit -m "feat: add expert prediction scale diagnostics"
```

---

### Task 4: Verify Full Test Suite for Touched Areas

**Files:**
- No file edits expected.

- [ ] **Step 1: Run focused pytest**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py tests/test_quitobench_expert_prediction_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 2: Run import smoke for runner CLI**

Run:

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py --help >/tmp/stage14e_runner_help.txt
```

Expected: exit code 0 and help includes `--train-set-standardize`.

- [ ] **Step 3: Commit only if verification required formatting or small fixes**

If fixes were needed:

```bash
git add tools/quitobench_framework_expert_cache.py tools/quitobench_expert_prediction_diagnostics.py tests/test_quitobench_dlinear_expert_cache.py tests/test_quitobench_expert_prediction_diagnostics.py
git commit -m "test: verify stage 1.4e expert alignment tools"
```

If no fixes were needed, do not create an empty commit.

---

### Task 5: Run Stage 1.4e Smoke and Official-Aligned Sanity Matrix

**Files:**
- Generated outputs under `outputs/vision_ts_routing/expert_predictions/...`
- Generated diagnostics JSON files under each relevant cache directory or `outputs/vision_ts_routing/expert_diagnostics/...`

- [ ] **Step 1: Run a 512-row DLinear smoke with scaler and scheduler**

Run:

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14e_scaler_smoke_512 \
  --stratified-rows 512 \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --device cuda
```

Expected: output dir printed, `manifest.json` records `standardization`, `scheduler=cosine`, and `drop_last=true`.

- [ ] **Step 2: Run a 512-row PatchTST smoke with scaler and scheduler**

Run:

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model patchtst \
  --expert-set-id patchtst_v1__stage14e_scaler_smoke_512 \
  --stratified-rows 512 \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --device cuda
```

Expected: output dir printed, predictions finite, no extreme `1e9+` prediction max in diagnostics.

- [ ] **Step 3: Run diagnostics on smoke caches**

Run diagnostics once per smoke cache:

```bash
conda run -n quito python tools/quitobench_expert_prediction_diagnostics.py \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stage14e_scaler_smoke_512 \
  --output-json outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stage14e_scaler_smoke_512/prediction_diagnostics.json
```

Repeat for PatchTST.

Expected: JSON files written with prediction/target/error scale summaries.

- [ ] **Step 4: Run 20k current-task official-aligned DLinear**

Run on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler \
  --stratified-rows 20000 \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --num-workers 6 \
  --device cuda
```

Expected: cache written and manifest records 20 epochs.

- [ ] **Step 5: Run 20k current-task official-aligned PatchTST**

Run on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model patchtst \
  --expert-set-id patchtst_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler \
  --stratified-rows 20000 \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --num-workers 6 \
  --device cuda
```

Expected: cache written and manifest records 20 epochs.

- [ ] **Step 6: Compare seasonal naive, DLinear, and PatchTST current-task caches**

Run:

```bash
conda run -n quito python tools/quitobench_expert_cache_comparison.py \
  --expert-set-ids lightweight_v1__seasonal_naive_full,dlinear_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler,patchtst_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler
```

Expected: comparison output has 20k common windows and top1 rates for all three experts.

- [ ] **Step 7: Decide whether to create `96/48/S` registry**

If the current-task official-aligned PatchTST remains anomalous, inspect whether `tools/quitobench_window_registry.py` can generate a separate `seq_len=96,pred_len=48` sample set without disrupting the current registry. If yes, create a separate Stage 1.4e subtask and output directory. If not, document the limitation and do not force a partial official-grid claim.

---

### Task 6: Write Stage 1.4e Experiment Log

**Files:**
- Create: `experiment_logs/2026-06-07_HHMM_stage1_4e_patchtst_official_alignment.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: Create log with implementation and result sections**

Use the actual current time in `HHMM`. Include:

```markdown
# Stage 1.4e：PatchTST 官方口径对齐与尺度审计

## 1. 目的

说明本阶段只审计 PatchTST/DLinear 官方口径，不实现 router/gate，不运行视觉 encoder，不生成 OOF cache。

## 2. 代码变更

列出 scaler、scheduler、drop_last、diagnostics 工具和测试结果。

## 3. Smoke 结果

记录 512-row DLinear/PatchTST smoke 的 manifest、train loss、prediction diagnostics。

## 4. 20k sanity 结果

记录 DLinear/PatchTST 当前任务官方对齐训练结果和 comparison。

## 5. 结论

按 spec 判定标准写出 PatchTST 异常更可能来自哪类原因。

## 6. 下一步

说明是否继续创建 `96/48/S` registry，是否保留 PatchTST 进入 OOF 候选。
```

- [ ] **Step 2: Register log in overview**

Add one row to `experiment_logs/实验日志总览.md`. Replace the result and next-step cells with concrete values from the Stage 1.4e run, such as the best fixed expert, PatchTST MSE/top1, and whether `96/48/S` should be generated:

```markdown
| 2026-06-07 HH:MM | Stage 1.4e | `2026-06-07_HHMM_stage1_4e_patchtst_official_alignment.md` | 对 PatchTST 做官方口径对齐与尺度审计 | 已完成 | 记录 best fixed、PatchTST/DLinear MSE、oracle top1 和尺度诊断结论 | 记录是否生成 `96/48/S` registry、是否保留 PatchTST 进入 OOF 候选 |
```

- [ ] **Step 3: Commit log**

```bash
git add experiment_logs/2026-06-07_HHMM_stage1_4e_patchtst_official_alignment.md experiment_logs/实验日志总览.md
git commit -m "docs: report stage 1.4e patchtst alignment audit"
```

---

## Self-Review Checklist

- [ ] Every new CLI flag has a test.
- [ ] Standardized training still writes predictions in original target scale.
- [ ] Manifest records whether standardization was enabled and what approximation scope was used.
- [ ] Diagnostics can be run independently on existing cache outputs.
- [ ] Experiment log states that wrapper-level scaler is not a perfect Quito `TimeSeriesDataset` clone unless proven otherwise.
- [ ] No router/gate, visual encoder, OOF cache, or Quito upstream modification is introduced.
