# TimeFuse 视觉融合兼容性实现计划

> **给 agentic workers 的要求：** 执行本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。每一步使用 checkbox（`- [ ]`）追踪。

**目标：** 建立 TimeFuse 兼容性探针，依次完成 artifact 审计、expert matrix 导出、视觉 embedding smoke 和 fusion ablation，判断 TimeFuse 是否适合作为视觉 embedding 增益验证平台。

**架构：** 保持 `TimeFuse/` 作为外部上游仓库，不直接修改其源码。新增本项目自有适配脚本到 `tools/`，读取 TimeFuse `meta_data/*.h5`，输出到 `outputs/timefuse_visual_fusion/`，并用测试保证每个适配层可独立验证。

**技术栈：** Python、NumPy、pandas、h5py、PyTorch、pytest，以及现有 `tools/quitobench_common.py` 中的 manifest helper。

---

## 文件结构

- 新建 `tools/timefuse_common.py`：TimeFuse H5 读取、样本 ID、shape 校验、finite 校验、预测误差计算。
- 新建 `tools/timefuse_artifact_audit.py`：检查 TimeFuse `meta_data` 是否齐全、shape 是否正确、数组是否 finite。
- 新建 `tools/timefuse_matrix_export.py`：把 `y_pred/y_true` 导出为本项目 long-format expert prediction/error table。
- 新建 `tools/timefuse_visual_embedding_smoke.py`：把 TimeFuse history window 转成三视图 tensor，并用小 CNN 生成 smoke embedding。
- 新建 `tools/timefuse_fusion_ablation.py`：比较 `uniform`、`best single`、`meta-only`、`visual-only`、`meta+visual`。
- 新建对应测试：
  - `tests/test_timefuse_common.py`
  - `tests/test_timefuse_artifact_audit.py`
  - `tests/test_timefuse_matrix_export.py`
  - `tests/test_timefuse_visual_embedding_smoke.py`
  - `tests/test_timefuse_fusion_ablation.py`
- 新建实验日志：
  - `experiment_logs/2026-06-08_timefuse_compatibility_probe.md`

不修改：

- `TimeFuse/`
- 现有 QuitoBench 工具
- 现有 QuitoBench 输出

## Task 1：新增 TimeFuse 公共 helper

**文件：**
- 新建：`tools/timefuse_common.py`
- 新建测试：`tests/test_timefuse_common.py`

- [ ] **Step 1：先写失败测试**

创建 `tests/test_timefuse_common.py`：

```python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.timefuse_common import (
    TimeFuseArrayBundle,
    build_timefuse_sample_ids,
    compute_forecast_errors,
    ensure_finite_array,
    load_h5_array,
    validate_timefuse_bundle,
)


def _write_h5(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("arr", data=arr)


def test_load_h5_array_reads_arr_dataset(tmp_path: Path) -> None:
    path = tmp_path / "x.h5"
    _write_h5(path, np.arange(6, dtype=np.float32).reshape(2, 3))

    loaded = load_h5_array(path)

    assert loaded.shape == (2, 3)
    assert loaded.dtype == np.float32


def test_build_timefuse_sample_ids_are_stable() -> None:
    ids = build_timefuse_sample_ids("ETTh1", "test", 96, 48, 96, 3)

    assert ids == [
        "ETTh1__test__sl96_ll48_pl96__row000000",
        "ETTh1__test__sl96_ll48_pl96__row000001",
        "ETTh1__test__sl96_ll48_pl96__row000002",
    ]


def test_validate_timefuse_bundle_accepts_matching_shapes() -> None:
    bundle = TimeFuseArrayBundle(
        dataset="ETTh1",
        split="test",
        seq_len=96,
        label_len=48,
        pred_len=96,
        model_names=("DLinear", "PatchTST"),
        x_meta=np.ones((4, 5), dtype=np.float32),
        y_pred=np.ones((4, 2, 96, 7), dtype=np.float32),
        y_true=np.ones((4, 96, 7), dtype=np.float32),
    )

    summary = validate_timefuse_bundle(bundle)

    assert summary["num_samples"] == 4
    assert summary["num_models"] == 2
    assert summary["num_channels"] == 7


def test_validate_timefuse_bundle_rejects_model_count_mismatch() -> None:
    bundle = TimeFuseArrayBundle(
        dataset="ETTh1",
        split="test",
        seq_len=96,
        label_len=48,
        pred_len=96,
        model_names=("DLinear",),
        x_meta=np.ones((4, 5), dtype=np.float32),
        y_pred=np.ones((4, 2, 96, 7), dtype=np.float32),
        y_true=np.ones((4, 96, 7), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="model_names"):
        validate_timefuse_bundle(bundle)


def test_ensure_finite_array_rejects_nan() -> None:
    arr = np.array([1.0, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="not finite"):
        ensure_finite_array(arr, label="x_meta")


def test_compute_forecast_errors_returns_per_sample_model_errors() -> None:
    y_true = np.zeros((2, 3, 1), dtype=np.float32)
    y_pred = np.array(
        [
            [[[0.0], [1.0], [2.0]], [[1.0], [1.0], [1.0]]],
            [[[0.0], [0.0], [0.0]], [[2.0], [0.0], [0.0]]],
        ],
        dtype=np.float32,
    )

    mse, mae = compute_forecast_errors(y_pred, y_true)

    assert mse.shape == (2, 2)
    assert mae.shape == (2, 2)
    assert np.allclose(mse[0], [5.0 / 3.0, 1.0])
    assert np.allclose(mae[1], [0.0, 2.0 / 3.0])
```

- [ ] **Step 2：确认测试失败**

运行：

```bash
pytest tests/test_timefuse_common.py -v
```

预期：因为 `tools.timefuse_common` 尚不存在，测试失败。

- [ ] **Step 3：实现最小公共 helper**

创建 `tools/timefuse_common.py`：

```python
"""TimeFuse 兼容性工具的公共 helper。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


DEFAULT_TIMEFUSE_ROOT = Path(__file__).resolve().parents[1] / "TimeFuse"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "timefuse_visual_fusion"


@dataclass(frozen=True)
class TimeFuseArrayBundle:
    dataset: str
    split: str
    seq_len: int
    label_len: int
    pred_len: int
    model_names: Sequence[str]
    x_meta: np.ndarray
    y_pred: np.ndarray
    y_true: np.ndarray


def load_h5_array(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少 H5 数组：{path}")
    with h5py.File(path, "r") as handle:
        if "arr" not in handle:
            raise ValueError(f"H5 文件缺少 arr dataset：{path}")
        return handle["arr"][:]


def ensure_finite_array(arr: np.ndarray, label: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains not finite values")


def build_timefuse_sample_ids(
    dataset: str,
    split: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    num_samples: int,
) -> list[str]:
    prefix = f"{dataset}__{split}__sl{seq_len}_ll{label_len}_pl{pred_len}"
    return [f"{prefix}__row{idx:06d}" for idx in range(int(num_samples))]


def validate_timefuse_bundle(bundle: TimeFuseArrayBundle) -> dict[str, int | str]:
    ensure_finite_array(bundle.x_meta, "x_meta")
    ensure_finite_array(bundle.y_pred, "y_pred")
    ensure_finite_array(bundle.y_true, "y_true")
    if bundle.x_meta.ndim != 2:
        raise ValueError(f"x_meta must be [N,F], got {bundle.x_meta.shape}")
    if bundle.y_pred.ndim != 4:
        raise ValueError(f"y_pred must be [N,K,T,C], got {bundle.y_pred.shape}")
    if bundle.y_true.ndim != 3:
        raise ValueError(f"y_true must be [N,T,C], got {bundle.y_true.shape}")
    n_pred, k, pred_len, channels = bundle.y_pred.shape
    n_true, true_pred_len, true_channels = bundle.y_true.shape
    x_meta = bundle.x_meta
    if len(x_meta) < n_pred:
        raise ValueError("x_meta 行数少于 y_pred 样本数")
    if len(x_meta) > n_pred:
        x_meta = x_meta[:n_pred]
    if n_pred != n_true:
        raise ValueError(f"y_pred/y_true 样本数不一致：{n_pred} vs {n_true}")
    if pred_len != int(bundle.pred_len) or true_pred_len != int(bundle.pred_len):
        raise ValueError(f"pred_len 不一致：expected={bundle.pred_len}, y_pred={pred_len}, y_true={true_pred_len}")
    if channels != true_channels:
        raise ValueError(f"channel 数不一致：{channels} vs {true_channels}")
    if k != len(tuple(bundle.model_names)):
        raise ValueError(f"model_names 数量 {len(tuple(bundle.model_names))} 与 y_pred K={k} 不一致")
    return {
        "dataset": bundle.dataset,
        "split": bundle.split,
        "num_samples": int(n_pred),
        "num_models": int(k),
        "pred_len": int(pred_len),
        "num_channels": int(channels),
        "x_meta_dim": int(x_meta.shape[1]),
    }


def compute_forecast_errors(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_pred.ndim != 4 or y_true.ndim != 3:
        raise ValueError("y_pred must be [N,K,T,C] and y_true must be [N,T,C]")
    diff = y_pred - y_true[:, None, :, :]
    mse = np.mean(np.square(diff), axis=(2, 3))
    mae = np.mean(np.abs(diff), axis=(2, 3))
    return mse.astype(np.float64), mae.astype(np.float64)
```

- [ ] **Step 4：运行测试**

运行：

```bash
pytest tests/test_timefuse_common.py -v
```

预期：全部通过。

- [ ] **Step 5：提交 Task 1**

运行：

```bash
git add tools/timefuse_common.py tests/test_timefuse_common.py
git commit -m "feat: add TimeFuse shared array helpers"
```

## Task 2：实现 TimeFuse artifact 审计

**文件：**
- 新建：`tools/timefuse_artifact_audit.py`
- 新建测试：`tests/test_timefuse_artifact_audit.py`

- [ ] **Step 1：先写失败测试**

创建 `tests/test_timefuse_artifact_audit.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tools.timefuse_artifact_audit import audit_timefuse_artifacts


def _write_h5(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("arr", data=arr)


def test_audit_timefuse_artifacts_writes_summary(tmp_path: Path) -> None:
    root = tmp_path / "TimeFuse"
    meta = root / "meta_data" / "ETTh1_test"
    _write_h5(meta / "x_meta_96.h5", np.ones((3, 4), dtype=np.float32))
    _write_h5(meta / "y_pred_96_48_96.h5", np.ones((3, 2, 96, 7), dtype=np.float32))
    _write_h5(meta / "y_true_96_48_96.h5", np.ones((3, 96, 7), dtype=np.float32))
    output_dir = tmp_path / "out"

    manifest = audit_timefuse_artifacts(
        timefuse_root=root,
        output_dir=output_dir,
        datasets=["ETTh1"],
        splits=["test"],
        forecast_setting=(96, 48, 96),
        model_names=["DLinear", "PatchTST"],
    )

    assert manifest["stage"] == "timefuse_artifact_audit"
    assert manifest["all_passed"] is True
    assert manifest["bundle_count"] == 1
    assert (output_dir / "artifact_audit_summary.csv").exists()
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_audit_timefuse_artifacts_records_missing_files(tmp_path: Path) -> None:
    manifest = audit_timefuse_artifacts(
        timefuse_root=tmp_path / "missing",
        output_dir=tmp_path / "out",
        datasets=["ETTh1"],
        splits=["test"],
        forecast_setting=(96, 48, 96),
        model_names=["DLinear", "PatchTST"],
    )

    assert manifest["all_passed"] is False
    assert manifest["missing_bundle_count"] == 1
```

- [ ] **Step 2：确认测试失败**

运行：

```bash
pytest tests/test_timefuse_artifact_audit.py -v
```

预期：因为模块尚不存在，测试失败。

- [ ] **Step 3：实现 artifact audit**

创建 `tools/timefuse_artifact_audit.py`。实现要求：

- 支持纯函数 `audit_timefuse_artifacts(...)`。
- 支持 CLI 参数：`--timefuse-root`、`--output-dir`、`--datasets`、`--splits`、`--forecast-setting`、`--model-names`。
- 输出：
  - `artifact_audit_summary.csv`
  - `manifest.json`
- 缺文件时不抛异常，记录到 summary，并让 `all_passed=False`。
- shape 或 finite 校验失败时记录错误信息。

核心逻辑：

```python
def _bundle_paths(timefuse_root: Path, dataset: str, split: str, seq_len: int, label_len: int, pred_len: int) -> tuple[Path, Path, Path]:
    base = Path(timefuse_root) / "meta_data" / f"{dataset}_{split}"
    return (
        base / f"x_meta_{seq_len}.h5",
        base / f"y_pred_{seq_len}_{label_len}_{pred_len}.h5",
        base / f"y_true_{seq_len}_{label_len}_{pred_len}.h5",
    )
```

- [ ] **Step 4：运行测试和 CLI help**

运行：

```bash
pytest tests/test_timefuse_artifact_audit.py tests/test_timefuse_common.py -v
python tools/timefuse_artifact_audit.py --help
```

预期：测试全部通过；help 输出包含 `--timefuse-root`。

- [ ] **Step 5：提交 Task 2**

运行：

```bash
git add tools/timefuse_artifact_audit.py tests/test_timefuse_artifact_audit.py
git commit -m "feat: add TimeFuse artifact audit"
```

## Task 3：导出 TimeFuse expert matrix

**文件：**
- 新建：`tools/timefuse_matrix_export.py`
- 新建测试：`tests/test_timefuse_matrix_export.py`

- [ ] **Step 1：先写失败测试**

创建 `tests/test_timefuse_matrix_export.py`，用 toy `y_pred/y_true` 验证：

- 写出 `predictions.parquet`
- 写出 `errors.parquet`
- 写出 `x_meta.parquet`
- 写出 `manifest.json`
- `(timefuse_sample_id, expert_id)` 不重复
- MSE/MAE 与手算结果一致

关键断言：

```python
assert len(preds) == 4
assert len(errors) == 4
assert preds[["timefuse_sample_id", "expert_id"]].duplicated().sum() == 0
assert set(preds.columns) >= {"yhat_0", "yhat_1", "target_0", "target_1"}
assert errors.loc[errors["expert_id"] == "A", "mse"].tolist() == [0.5, 0.0]
```

- [ ] **Step 2：确认测试失败**

运行：

```bash
pytest tests/test_timefuse_matrix_export.py -v
```

预期：因为模块尚不存在，测试失败。

- [ ] **Step 3：实现 matrix export**

创建 `tools/timefuse_matrix_export.py`。实现要求：

- 读取 `x_meta/y_pred/y_true`。
- 复用 `TimeFuseArrayBundle` 和 `validate_timefuse_bundle`。
- 生成稳定 `timefuse_sample_id`。
- 第一版使用 `channel_index="all"`，在 `[pred_len, C]` 上计算整体 MSE/MAE。
- 对 wide prediction table，先对 channel 做 mean，生成 `yhat_0 ... yhat_{pred_len-1}` 和 `target_0 ... target_{pred_len-1}`。

核心 row-building 函数：

```python
def _build_prediction_and_error_tables(bundle: TimeFuseArrayBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_timefuse_bundle(bundle)
    sample_ids = build_timefuse_sample_ids(
        bundle.dataset,
        bundle.split,
        bundle.seq_len,
        bundle.label_len,
        bundle.pred_len,
        bundle.y_pred.shape[0],
    )
    mse, mae = compute_forecast_errors(bundle.y_pred, bundle.y_true)
    pred_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for sample_idx, sample_id in enumerate(sample_ids):
        target_flat = bundle.y_true[sample_idx].mean(axis=1)
        for model_idx, expert_id in enumerate(bundle.model_names):
            pred_flat = bundle.y_pred[sample_idx, model_idx].mean(axis=1)
            pred_row = {
                "timefuse_sample_id": sample_id,
                "dataset": bundle.dataset,
                "split": bundle.split,
                "seq_len": int(bundle.seq_len),
                "label_len": int(bundle.label_len),
                "pred_len": int(bundle.pred_len),
                "channel_index": "all",
                "expert_id": str(expert_id),
            }
            for horizon_idx, value in enumerate(pred_flat):
                pred_row[f"yhat_{horizon_idx}"] = float(value)
            for horizon_idx, value in enumerate(target_flat):
                pred_row[f"target_{horizon_idx}"] = float(value)
            pred_rows.append(pred_row)
            error_rows.append(
                {
                    "timefuse_sample_id": sample_id,
                    "dataset": bundle.dataset,
                    "split": bundle.split,
                    "channel_index": "all",
                    "expert_id": str(expert_id),
                    "mse": float(mse[sample_idx, model_idx]),
                    "mae": float(mae[sample_idx, model_idx]),
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(error_rows)
```

- [ ] **Step 4：运行测试**

运行：

```bash
pytest tests/test_timefuse_matrix_export.py tests/test_timefuse_common.py -v
```

预期：全部通过。

- [ ] **Step 5：提交 Task 3**

运行：

```bash
git add tools/timefuse_matrix_export.py tests/test_timefuse_matrix_export.py
git commit -m "feat: export TimeFuse expert matrix"
```

## Task 4：实现视觉 embedding smoke

**文件：**
- 新建：`tools/timefuse_visual_embedding_smoke.py`
- 新建测试：`tests/test_timefuse_visual_embedding_smoke.py`

- [ ] **Step 1：先写失败测试**

创建 `tests/test_timefuse_visual_embedding_smoke.py`，验证：

- `build_timefuse_view_tensor(histories)` 输出 `[N,3,H,W]`。
- view tensor 全 finite，范围在 `[0,1]`。
- `TimeFuseTinyVisualEncoder` 输出指定维度 embedding。
- `encode_timefuse_views` 同 seed 下确定性一致。
- `build_timefuse_embedding_table` 保留 `timefuse_sample_id`。

核心断言：

```python
assert views.shape == (4, 3, 16, 32)
assert np.isfinite(views).all()
assert views.min() >= 0.0
assert views.max() <= 1.0
assert embeddings.shape == (3, 8)
```

- [ ] **Step 2：确认测试失败**

运行：

```bash
pytest tests/test_timefuse_visual_embedding_smoke.py -v
```

预期：因为模块尚不存在，测试失败。

- [ ] **Step 3：实现视觉 smoke**

创建 `tools/timefuse_visual_embedding_smoke.py`。实现要求：

- 输入 `histories` 形状为 `[N,T,C]`。
- 第一版对 channel 做 mean，构造三视图：
  - line raster
  - period fold
  - fft power
- 使用小 CNN `TimeFuseTinyVisualEncoder`，不训练视觉 encoder。
- 输出 embedding table：`timefuse_sample_id`、`encoder_id`、`z_*`。

核心函数签名：

```python
def build_timefuse_view_tensor(histories: np.ndarray, height: int = 64, width: int = 192) -> np.ndarray:
    ...


class TimeFuseTinyVisualEncoder(nn.Module):
    ...


def encode_timefuse_views(
    views: np.ndarray,
    embedding_dim: int = 64,
    random_seed: int = 20260608,
    batch_size: int = 128,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, object]]:
    ...


def build_timefuse_embedding_table(sample_ids: list[str], embeddings: torch.Tensor, encoder_id: str) -> pd.DataFrame:
    ...
```

- [ ] **Step 4：运行测试**

运行：

```bash
pytest tests/test_timefuse_visual_embedding_smoke.py -v
```

预期：全部通过。

- [ ] **Step 5：提交 Task 4**

运行：

```bash
git add tools/timefuse_visual_embedding_smoke.py tests/test_timefuse_visual_embedding_smoke.py
git commit -m "feat: add TimeFuse visual embedding smoke"
```

## Task 5：实现 fusion ablation

**文件：**
- 新建：`tools/timefuse_fusion_ablation.py`
- 新建测试：`tests/test_timefuse_fusion_ablation.py`

- [ ] **Step 1：先写失败测试**

创建 `tests/test_timefuse_fusion_ablation.py`，验证：

- `compute_best_single_and_uniform` 能计算 best single、uniform 和 oracle top1。
- `train_linear_fusor_ablation` 返回三行：`meta_only`、`visual_only`、`meta_visual`。
- 三个 learned baseline 使用同一训练函数，只改变输入特征矩阵。

核心断言：

```python
assert scores["best_single_model"] == "A"
assert scores["best_single_mse"] == 0.5
assert scores["uniform_mse"] == 0.25
assert set(result["baseline"]) == {"meta_only", "visual_only", "meta_visual"}
assert result["mse"].notna().all()
assert result["mae"].notna().all()
```

- [ ] **Step 2：确认测试失败**

运行：

```bash
pytest tests/test_timefuse_fusion_ablation.py -v
```

预期：因为模块尚不存在，测试失败。

- [ ] **Step 3：实现 fusion ablation**

创建 `tools/timefuse_fusion_ablation.py`。实现要求：

- `best single` 和 `uniform` 不训练。
- `meta_only`、`visual_only`、`meta_visual` 使用相同 `_train_and_eval_fusor`。
- fusor 为 `nn.Linear(input_dim, num_models)` + `softmax`。
- weighted prediction 为 `weights[:, :, None, None] * y_pred` 后沿 model 维求和。
- 输出列：
  - `baseline`
  - `mse`
  - `mae`
  - `weight_entropy`
  - `best_single_mse`
  - `uniform_mse`
  - `oracle_top1_mse`

核心 best/uniform 函数：

```python
def compute_best_single_and_uniform(y_pred: np.ndarray, y_true: np.ndarray, model_names: Sequence[str]) -> dict[str, object]:
    diff = y_pred - y_true[:, None, :, :]
    per_model_mse = np.mean(np.square(diff), axis=(0, 2, 3))
    per_model_mae = np.mean(np.abs(diff), axis=(0, 2, 3))
    best_idx = int(np.argmin(per_model_mse))
    uniform_pred = np.mean(y_pred, axis=1)
    uniform_diff = uniform_pred - y_true
    oracle_mse = np.mean(np.min(np.mean(np.square(diff), axis=(2, 3)), axis=1))
    return {
        "best_single_model": str(model_names[best_idx]),
        "best_single_mse": float(per_model_mse[best_idx]),
        "best_single_mae": float(per_model_mae[best_idx]),
        "uniform_mse": float(np.mean(np.square(uniform_diff))),
        "uniform_mae": float(np.mean(np.abs(uniform_diff))),
        "oracle_top1_mse": float(oracle_mse),
    }
```

- [ ] **Step 4：运行测试**

运行：

```bash
pytest tests/test_timefuse_fusion_ablation.py -v
```

预期：全部通过。

- [ ] **Step 5：提交 Task 5**

运行：

```bash
git add tools/timefuse_fusion_ablation.py tests/test_timefuse_fusion_ablation.py
git commit -m "feat: add TimeFuse fusion ablation"
```

## Task 6：端到端 smoke 与实验日志

**文件：**
- 新建：`experiment_logs/2026-06-08_timefuse_compatibility_probe.md`
- 修改：`experiment_logs/实验日志总览.md`

- [ ] **Step 1：运行聚焦测试**

运行：

```bash
pytest tests/test_timefuse_common.py tests/test_timefuse_artifact_audit.py tests/test_timefuse_matrix_export.py tests/test_timefuse_visual_embedding_smoke.py tests/test_timefuse_fusion_ablation.py -v
```

预期：全部通过。

- [ ] **Step 2：对当前 TimeFuse clone 运行 artifact audit**

运行：

```bash
python tools/timefuse_artifact_audit.py --timefuse-root TimeFuse --output-dir outputs/timefuse_visual_fusion/artifact_audit
```

预期：命令退出码为 0，写出 `outputs/timefuse_visual_fusion/artifact_audit/manifest.json`；在还没有下载 Google Drive 数据包前，`all_passed` 应为 `False`。

- [ ] **Step 3：写实验日志**

创建 `experiment_logs/2026-06-08_timefuse_compatibility_probe.md`：

```markdown
# 2026-06-08 TimeFuse 兼容性探针

## 目的

验证 TimeFuse 是否适合作为视觉 embedding 对专家融合增益的主验证平台。

## 已完成

- 已克隆 `ZhiningLiu1998/TimeFuse` 到 `TimeFuse/`。
- 已确认 TimeFuse 核心协议为 `x_meta / y_model_preds / y_true`。
- 已新增 TimeFuse artifact audit、expert matrix export、visual embedding smoke 和 fusion ablation 适配入口。

## 验证命令

```bash
pytest tests/test_timefuse_common.py tests/test_timefuse_artifact_audit.py tests/test_timefuse_matrix_export.py tests/test_timefuse_visual_embedding_smoke.py tests/test_timefuse_fusion_ablation.py -v
python tools/timefuse_artifact_audit.py --timefuse-root TimeFuse --output-dir outputs/timefuse_visual_fusion/artifact_audit
```

## 当前结论

TimeFuse 代码接口适合迁移视觉融合验证，但当前 clone 不包含 README 所述的 `dataset/`、`meta_data/`、`checkpoints/`。下一步需要下载 TimeFuse 数据包后再运行 matrix export 和 fusion ablation。

## 下一步

下载 TimeFuse README 指向的数据包，复跑 artifact audit。若 `all_passed=True`，进入 expert matrix 导出与 visual embedding smoke。
```

- [ ] **Step 4：更新实验日志总览**

在 `experiment_logs/实验日志总览.md` 追加：

```markdown
| 2026-06-08 | TimeFuse 兼容性 | `2026-06-08_timefuse_compatibility_probe.md` | 验证 TimeFuse 是否适合作为视觉 embedding 对专家融合增益的主验证平台 | 已完成 | 已建立 TimeFuse artifact audit、expert matrix export、visual embedding smoke 和 fusion ablation 适配入口；当前 clone 缺少 README 数据包 | 下载 TimeFuse 数据包后复跑 artifact audit，并进入 matrix export |
```

- [ ] **Step 5：提交 Task 6**

运行：

```bash
git add experiment_logs/2026-06-08_timefuse_compatibility_probe.md experiment_logs/实验日志总览.md
git commit -m "docs: log TimeFuse compatibility probe"
```

## 自检

- 规格覆盖：本计划覆盖 artifact 审计、expert matrix 映射、visual embedding 对齐、fusion ablation、输出目录隔离、依赖风险和实验日志。
- 文档完整性：计划中每个任务都有测试、实现、验证和提交步骤。
- 命名一致性：样本 ID 字段统一为 `timefuse_sample_id`；forecast setting 顺序统一为 `(seq_len, label_len, pred_len)`；预测 tensor 形状统一为 `[N, K, pred_len, C]`。
