# Stage 1.1b Torch Light Proxy Online Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Stage 1.1 light proxy 增加 torch batch kernel，支持在线 CPU/GPU 计算并与 NumPy reference 保持一致。

**Architecture:** 在 `tools/quitobench_sample_channel_light_proxy.py` 中保留现有 NumPy/Pandas 离线预计算路径，新增 `compute_light_proxy_torch(histories, periods) -> torch.Tensor` 作为在线 kernel。测试以现有 `compute_window_proxy()` 和 `FEATURE_COLUMNS` 为 reference，确保 feature order、数值、device 行为和无 Pandas 依赖。

**Tech Stack:** Python 3.11, torch, numpy, pytest, conda env `quito`.

---

### Task 1: Torch Kernel Tests

**Files:**
- Modify: `tests/test_quitobench_sample_channel_light_proxy.py`
- Modify after red: `tools/quitobench_sample_channel_light_proxy.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
from tools.quitobench_sample_channel_light_proxy import (
    FEATURE_COLUMNS,
    compute_light_proxy_torch,
    compute_window_proxy,
)
```

Required behavior:

- `compute_light_proxy_torch(histories, periods)` accepts `[B, L]`.
- Output shape is `[B, len(FEATURE_COLUMNS)]`.
- Feature order matches `FEATURE_COLUMNS`.
- CPU output matches NumPy `compute_window_proxy()` on toy data.
- If CUDA is available, CUDA output stays on CUDA and matches CPU output.

- [ ] **Step 2: Verify red**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
```

Expected: fail because `compute_light_proxy_torch` is missing.

- [ ] **Step 3: Implement kernel**

Add torch import and implement:

```python
def compute_light_proxy_torch(histories: torch.Tensor, periods: torch.Tensor | Sequence[int]) -> torch.Tensor:
    ...
```

Constraints:

- Do not import or use pandas in the kernel.
- Do not accept target/future inputs.
- Preserve input device.
- Return finite float tensor.
- Feature order exactly follows `FEATURE_COLUMNS`.

- [ ] **Step 4: Verify green**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
conda run -n quito python -m pytest tests -q
```

Expected: all tests pass.

---

### Task 2: Stage 1.1b Log and Commit

**Files:**
- Create: `experiment_logs/YYYY-MM-DD_HHMM_stage1_1b_torch_light_proxy_kernel.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: Run minimal CPU/GPU verification**

Run a Python one-liner or pytest that prints:

```text
torch cuda available
cpu tensor shape
cuda tensor shape
max abs parity error
```

- [ ] **Step 2: Write Chinese experiment log**

Record:

- purpose
- reference implementation
- feature order
- CPU/GPU verification
- no cache recomputation
- no router/expert execution

- [ ] **Step 3: Update overview**

Append a Stage 1.1b row.

- [ ] **Step 4: Commit**

Run:

```bash
git add tools/quitobench_sample_channel_light_proxy.py tests/test_quitobench_sample_channel_light_proxy.py docs/superpowers/plans/2026-06-07-stage1-1b-torch-light-proxy.md experiment_logs/<log>.md experiment_logs/实验日志总览.md
git commit -m "feat: add torch light proxy kernel"
```
