# Stage 0.7 Channel STL Codebook Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 QuitoBench 通道级 full-length STL 质量 CSV，并用论文 channel-mean + `tau=0.4` 口径验证 Stage 0.6b 官方 codebook。

**Architecture:** 新增一个 Stage 0.7 独立脚本，沿用现有 `tools/` 轻量实验脚本结构。脚本先写可 resume 的通道级中间 CSV，再聚合 item 级均值并生成官方 codebook 验证表和中文报告。

**Tech Stack:** Python, pandas, numpy, pytest, Quito `evaluate_series`, conda env `quito`

---

### Task 1: 写测试锁定 Stage 0.7 数据规则

**Files:**
- Create: `tests/test_quitobench_channel_stl_codebook_validation.py`
- Create: `tools/quitobench_channel_stl_codebook_validation.py`

- [ ] **Step 1: 写失败测试**

测试内容：

```python
def test_filter_completed_channel_tasks_uses_subset_item_channel_key() -> None:
    tasks = [
        {"subset": "hour", "item_id": 1, "channel": "ind_1"},
        {"subset": "hour", "item_id": 1, "channel": "ind_2"},
        {"subset": "min", "item_id": 1, "channel": "ind_1"},
    ]
    existing = pd.DataFrame([{"subset": "hour", "item_id": 1, "channel": "ind_1"}])
    assert filter_completed_channel_tasks(tasks, existing) == [
        {"subset": "hour", "item_id": 1, "channel": "ind_2"},
        {"subset": "min", "item_id": 1, "channel": "ind_1"},
    ]
```

同时覆盖：

- `cell_from_thresholds()` 使用 `> 0.4` 为 high。
- `build_item_channel_mean()` 对 5 个 channel 三项指标求均值。
- `compare_with_official_codebook()` 生成 exact/dim match 字段。
- `build_confusion_matrix()` 生成官方 cell x paper-like cell item 计数。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_channel_stl_codebook_validation.py -q
```

Expected: import 或函数缺失失败。

### Task 2: 实现 Stage 0.7 脚本

**Files:**
- Modify: `tools/quitobench_channel_stl_codebook_validation.py`

- [ ] **Step 1: 实现任务生成与通道评估**

实现：

- `ChannelQualityTask`
- `iter_channel_tasks()`
- `evaluate_channel_task()`
- `read_existing_channel_quality()`
- `filter_completed_channel_tasks()`
- `write_channel_quality_csv()`
- `compute_channel_quality_with_progress()`

- [ ] **Step 2: 实现聚合和验证**

实现：

- `cell_from_thresholds()`
- `build_item_channel_mean()`
- `compare_with_official_codebook()`
- `build_cluster_summary()`
- `build_confusion_matrix()`
- `write_validation_report()`

- [ ] **Step 3: 实现 CLI**

默认参数：

```bash
--data-dir data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315
--output-dir outputs/data_audit
--codebook outputs/data_audit/quitobench_official_cluster_codebook.csv
--tau 0.4
--max-workers 8
--batch-size 25
```

### Task 3: 运行测试并做小样本 smoke

**Files:**
- Modify if needed: `tools/quitobench_channel_stl_codebook_validation.py`
- Modify if needed: `tests/test_quitobench_channel_stl_codebook_validation.py`

- [ ] **Step 1: 运行单元测试**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_channel_stl_codebook_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 2: 运行小样本 smoke**

Run:

```bash
conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py --max-items-per-subset 2 --max-workers 1 --batch-size 2 --output-dir outputs/data_audit/stage0_7_smoke
```

Expected: 生成 4 个输出文件，通道级 CSV 有 20 行。

### Task 4: 启动全量长实验并记录日志

**Files:**
- Create: `experiment_logs/2026-06-06_1950_stage0_7_channel_stl_codebook_validation.md`
- Modify: `experiment_logs/实验日志总览.md`

- [ ] **Step 1: 写实验日志初稿**

包含目的、计划、命令、输入配置、预期输出、resume 策略。

- [ ] **Step 2: 启动全量命令**

Run:

```bash
conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py --max-workers 8 --batch-size 25
```

Expected: 分批打印进度，持续写 `outputs/data_audit/quitobench_channel_quality_stl_full.csv`。

- [ ] **Step 3: 实验结束后写结果**

记录输出文件行数、match 指标、cluster 24 解释、问题观察、下一步。

### Task 5: 完成验证

- [ ] **Step 1: 运行完整性检查**

Run:

```bash
python3 - <<'PY'
import pandas as pd
paths = [
    'outputs/data_audit/quitobench_channel_quality_stl_full.csv',
    'outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv',
    'outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv',
]
for p in paths:
    df = pd.read_csv(p)
    print(p, df.shape)
PY
wc -l outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md
```

Expected:

- 通道级 CSV 约 6,450 行加表头。
- item 级和验证 CSV 为 1,290 行加表头。
- report 存在且非空。

- [ ] **Step 2: 更新日志总览**

新增 Stage 0.7 行，状态按实际结果填写。

## 自检

- 覆盖设计中的全部输出文件。
- 明确使用 QuitoBench benchmark，不使用 Quito corpus。
- 明确 Stage 0.7 不替代官方 codebook。
- 支持 resume，适合长实验中断后续跑。
- 测试覆盖核心纯数据逻辑，长耗时 `evaluate_series` 只在 smoke/full run 中执行。
