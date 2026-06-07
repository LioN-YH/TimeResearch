# Stage 1.2b Proxy + Imageization Latency Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 系统测量 Stage 1.1b torch proxy kernel 与 Stage 1.2 三视图 imageization 在线路径在不同 batch size、CPU/GPU 下的成本。

**Architecture:** 新增 `tools/quitobench_proxy_imageization_latency.py`，复用 `compute_light_proxy_torch()`、`imageize_batch()` 和 Stage 1.2 的 registry history 切片函数。脚本只做抽样、warmup/repeat 计时和 CSV/manifest 写出，不重新计算 Stage 1.1 cache，不运行专家模型，不实现 router。

**Tech Stack:** Python 3.11, pandas, numpy, torch, pytest, conda env `quito`.

---

### Task 1: Latency Sweep Core

**Files:**
- Create: `tests/test_quitobench_proxy_imageization_latency.py`
- Create: `tools/quitobench_proxy_imageization_latency.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
from tools.quitobench_proxy_imageization_latency import (
    LatencySweepConfig,
    benchmark_online_components,
    build_latency_manifest,
    write_latency_outputs,
)
```

Required behavior:

- `benchmark_online_components()` accepts `[N, L]` histories, periods, device, and batch size.
- Output row includes `proxy_torch_latency_ms_per_window`、`view_tensor_latency_ms_per_window`、`proxy_plus_view_latency_ms_per_window`。
- The function does not read proxy cache or future target.
- `write_latency_outputs()` writes `stage1_2b_proxy_imageization_latency.csv` and `stage1_2b_proxy_imageization_latency_manifest.json`。

- [ ] **Step 2: Verify red**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_proxy_imageization_latency.py -q
```

Expected: fail because `tools.quitobench_proxy_imageization_latency` is missing.

- [ ] **Step 3: Implement minimal core and writer**

Implement:

```python
@dataclass(frozen=True)
class LatencySweepConfig:
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128, 512, 1024)
    devices: tuple[str, ...] = ("cpu", "cuda")
    warmup_iters: int = 3
    measure_iters: int = 10
```

`benchmark_online_components()` should time proxy alone, view tensor alone, and proxy + view sequentially, returning per-window latency.

- [ ] **Step 4: Verify green**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_proxy_imageization_latency.py -q
```

Expected: pass.

### Task 2: CLI Sweep, Logs, and Verification

**Files:**
- Modify: `tools/quitobench_proxy_imageization_latency.py`
- Create: `experiment_logs/YYYY-MM-DD_HHMM_stage1_2b_online_latency_sweep.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: Add CLI**

Arguments:

```text
--registry-dir
--data-dir
--output-dir
--batch-sizes
--devices
--warmup-iters
--measure-iters
--random-seed
```

- [ ] **Step 2: Run latency sweep**

Run:

```bash
conda run -n quito python tools/quitobench_proxy_imageization_latency.py
```

Expected outputs:

```text
outputs/vision_ts_routing/latency/stage1_2b_proxy_imageization_latency.csv
outputs/vision_ts_routing/latency/stage1_2b_proxy_imageization_latency_manifest.json
```

- [ ] **Step 3: Run tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_proxy_imageization_latency.py tests/test_quitobench_sample_channel_light_proxy.py tests/test_quitobench_imageization_protocol.py -q
```

- [ ] **Step 4: Write Chinese experiment log and update overview**

Record the commands, input sample set, CPU/GPU availability, CSV metrics, constraints, and next step. Append one Stage 1.2b row to `experiment_logs/实验日志总览.md`.
