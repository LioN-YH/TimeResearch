# Stage 1.2 Imageization Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 QuitoBench sample-channel history window 到三视图 `view_tensor [V,H,W]` 的 Stage 1.2 smoke 协议。

**Architecture:** 新增一个独立脚本 `tools/quitobench_imageization_protocol.py`，只读取 Stage 1.0 registry、原始 QuitoBench parquet 和 Stage 1.1 proxy manifest。核心 tensor 生成函数使用 `torch`，正式路径输出 `view_tensor_sample.npz`、`image_index.csv`、`manifest.json` 和少量 debug PNG，不训练 encoder、不运行专家、不实现 router。

**Tech Stack:** Python 3.11, pandas, numpy, torch, PIL/Pillow for sampled debug PNG only, pytest, conda env `quito`.

---

## File Structure

- Create `tools/quitobench_imageization_protocol.py`
  - 定义 `ImageizationConfig`。
  - 定义 `normalize_history_batch()`、`imageize_batch()`、`period_fold_view()`、`line_raster_view()`、`fft_power_view()`。
  - 定义 smoke 抽样、原始 history 切片、输出写入、debug PNG 保存、CLI。
- Create `tests/test_quitobench_imageization_protocol.py`
  - 覆盖 normalization、shape、period padding、stratified smoke sample、输出 manifest 和 proxy join。
- Create `experiment_logs/YYYY-MM-DD_HHMM_stage1_2_imageization_protocol_smoke.md`
  - 执行后记录目的、命令、输入、结果、问题、结论、下一步。
- Modify `experiment_logs/实验日志总览.md`
  - 登记 Stage 1.2 smoke。

---

### Task 1: Tensor Core Tests

**Files:**
- Create: `tests/test_quitobench_imageization_protocol.py`
- Create after red: `tools/quitobench_imageization_protocol.py`

- [ ] **Step 1: Write failing tests for normalization and tensor shape**

Add tests that import:

```python
from tools.quitobench_imageization_protocol import (
    ImageizationConfig,
    imageize_batch,
    normalize_history_batch,
)
```

Required assertions:

```python
def test_normalize_history_batch_is_per_window() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [100.0, 100.0, 100.0, 100.0]])
    normalized, meta = normalize_history_batch(x, ImageizationConfig(height=8, width=4))
    assert normalized.shape == x.shape
    assert meta["mean"].shape == (2,)
    assert meta["std"].shape == (2,)
    assert torch.isfinite(normalized).all()
    assert meta["mean"].tolist() == [2.5, 100.0]


def test_imageize_batch_outputs_three_view_tensor() -> None:
    x = torch.arange(384, dtype=torch.float32).reshape(2, 192)
    tensor, meta = imageize_batch(x, periods=[24, 144], config=ImageizationConfig())
    assert tensor.shape == (2, 3, 64, 192)
    assert meta["view_names"] == ["line_raster", "period_fold", "fft_power"]
    assert meta["padding_lengths"] == [0, 96]
    assert torch.isfinite(tensor).all()
```

- [ ] **Step 2: Run red test**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'tools.quitobench_imageization_protocol'`.

- [ ] **Step 3: Implement minimal tensor core**

Create `tools/quitobench_imageization_protocol.py` with:

```python
@dataclass(frozen=True)
class ImageizationConfig:
    height: int = 64
    width: int = 192
    norm_const: float = 0.4
    eps: float = 1e-5
    clip_min: float = -5.0
    clip_max: float = 5.0
    view_names: tuple[str, ...] = ("line_raster", "period_fold", "fft_power")
```

Implement functions using torch tensor ops. `imageize_batch()` must return `[B,3,H,W]` and metadata including `padding_lengths`.

- [ ] **Step 4: Run green test**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

Expected: pass current tests.

---

### Task 2: Smoke Sampling and Output Tests

**Files:**
- Modify: `tests/test_quitobench_imageization_protocol.py`
- Modify: `tools/quitobench_imageization_protocol.py`

- [ ] **Step 1: Write failing tests for stratified smoke sampling and output writing**

Required behavior:

- `sample_smoke_registry()` groups by `subset/split/official_tsf_cell` and samples at most `max_per_group`.
- `write_imageization_outputs()` writes:
  - `view_tensor_sample.npz`
  - `image_index.csv`
  - `manifest.json`
  - `debug_png/`
- `image_index.csv` keeps `physical_window_id` and `sample_set_id`.

- [ ] **Step 2: Run red test**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

Expected: fail because sampling/output functions are missing.

- [ ] **Step 3: Implement sampling and output writing**

Add:

```python
def sample_smoke_registry(registry: pd.DataFrame, max_per_group: int, random_seed: int) -> pd.DataFrame
def write_imageization_outputs(view_tensor: torch.Tensor, image_index: pd.DataFrame, manifest: Mapping[str, object], output_root: Path) -> Path
```

Use output directory:

```text
<sample_set_id>__stage1_2_smoke_v1
```

- [ ] **Step 4: Run green test**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

Expected: all Stage 1.2 tests pass.

---

### Task 3: CLI Smoke Implementation

**Files:**
- Modify: `tools/quitobench_imageization_protocol.py`

- [ ] **Step 1: Add CLI arguments**

Arguments:

```text
--registry-dir
--proxy-dir
--data-dir
--output-root
--max-per-group
--random-seed
--debug-png-count
--device
```

- [ ] **Step 2: Implement CLI flow**

Flow:

1. Read registry.
2. Stratified sample smoke rows.
3. Load needed subset parquet.
4. Slice only `[history_start_idx, history_end_idx)`.
5. Generate `view_tensor [N,V,H,W]`.
6. Validate proxy join by `physical_window_id`.
7. Write outputs and manifest.
8. Save sampled debug PNG from tensor only.

- [ ] **Step 3: Run smoke**

Run:

```bash
conda run -n quito python tools/quitobench_imageization_protocol.py --max-per-group 4 --debug-png-count 16 --device cpu
```

Expected:

- output directory under `outputs/vision_ts_routing/image_tensors/`
- tensor shape has `V=3,H=64,W=192`
- manifest says `runs_visual_encoder=false`, `runs_expert_models=false`, `implements_router=false`

---

### Task 4: Verification and Logs

**Files:**
- Create: `experiment_logs/YYYY-MM-DD_HHMM_stage1_2_imageization_protocol_smoke.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: Run full tests**

Run:

```bash
conda run -n quito python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Validate smoke output**

Run a Python check that reads `manifest.json`, `image_index.csv`, and `view_tensor_sample.npz`, then asserts:

```python
view_tensor.shape[1:] == (3, 64, 192)
len(image_index) == view_tensor.shape[0]
image_index["physical_window_id"].is_unique
manifest["view_tensor_semantics"] == "multi_view_not_rgb"
manifest["normalization"]["scope"] == "per_physical_window_id_history"
```

- [ ] **Step 3: Write experiment log**

Use Chinese sections:

```markdown
# Stage 1.2：伪图像协议与视觉输入 Smoke
## 1. 实验目的
## 2. 实验计划
## 3. 执行命令
## 4. 输入数据与配置
## 5. 实验结果
## 6. 问题与观察
## 7. 结论
## 8. 下一步计划
```

- [ ] **Step 4: Update overview**

Append one row to `experiment_logs/实验日志总览.md`.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/quitobench_imageization_protocol.py tests/test_quitobench_imageization_protocol.py docs/superpowers/plans/2026-06-07-stage1-2-imageization-protocol.md experiment_logs/<log>.md experiment_logs/实验日志总览.md
git commit -m "feat: add stage 1.2 imageization smoke"
```
