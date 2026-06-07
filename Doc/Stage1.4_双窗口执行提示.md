# Stage 1.4 双窗口执行提示

本文用于把 Stage 1.4 后续工作拆成两个互不阻塞的窗口执行。

## 1. 已创建的工作区

主工作区：

```text
/home/user10/TSF/DATAPrepare
```

窗口 A 工作区：

```text
/home/user10/TSF/DATAPrepare/.worktrees/stage1-4a-analysis
branch: stage1-4a-analysis
```

窗口 B 工作区：

```text
/home/user10/TSF/DATAPrepare/.worktrees/stage1-4b-framework
branch: stage1-4b-framework
```

两个 worktree 中已创建本地符号链接：

```text
data -> ../../data
outputs -> ../../outputs
```

因此脚本默认路径可以继续访问主目录中的 QuitoBench 数据和已有输出。

两个 worktree 的基线验证均已通过：

```bash
conda run -n quito python -m pytest tests -q
```

结果：

```text
45 passed
```

## 2. 通用上下文

项目根目录：

```text
/home/user10/TSF/DATAPrepare
```

当前最新主线提交：

```text
45b9cac chore: ignore local worktrees
```

最近关键提交：

```text
086b923 feat: add stage 1.4a lightweight expert cache
aba7ac1 docs: localize stage 1.4a plan
c24295a docs: plan stage 1.4a expert cache
04b551d docs: design stage 1.4 expert cache
```

必须阅读：

```text
Doc/视觉伪图像路由项目交接.md
Doc/视觉伪图像路由双路线实施计划.md
Doc/视觉路由实验路线对比_保守验证与推荐主线.md
docs/superpowers/specs/2026-06-07-stage1-4-expert-cache-framework-design.md
docs/superpowers/plans/2026-06-07-stage1-4a-lightweight-expert-cache.md
experiment_logs/2026-06-07_1151_stage1_4a_lightweight_expert_cache.md
experiment_logs/实验日志总览.md
```

当前 Stage 1.4a 已完成：

- 新增 `tools/quitobench_lightweight_expert_cache.py`
- 新增 `tests/test_quitobench_lightweight_expert_cache.py`
- smoke 输出：

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512/
```

Stage 1.4a smoke 结果：

```text
windows 512
prediction_rows 2048
error_rows 2048
prediction_unique True
error_unique True
soft_weight_max_abs_error 3.3306690738754696e-16
implements_router False
runs_neural_experts False
```

硬性约束：

- 使用 `conda run -n quito ...`。
- 文档、计划、日志、项目语义相关注释默认使用中文。
- 不实现 router / gate。
- 不运行视觉 encoder。
- Stage 1.4a-analysis 不接入神经网络专家。
- Stage 1.4b 可以审计并接入第一个正式训练型专家，但仍不实现 router。
- 每个实验都要写入 `experiment_logs/`，并更新 `experiment_logs/实验日志总览.md`。
- 输出目录必须使用不同 `expert_set_id`，不要互相覆盖。

## 3. 窗口 A：Stage 1.4a-expanded / analysis

### 目标

判断轻量专家是否值得作为最终专家池的一部分，而不是只作为工程占位。

### 建议任务

1. 基于当前 `tools/quitobench_lightweight_expert_cache.py`，先做分层抽样版本：

```text
expert_set_id = lightweight_v1__stratified_50k
```

不建议一开始直接跑全量 62.7 万。先按：

```text
split / subset / official_tsf_cell
```

分层抽样约 50k 窗口。

2. 输出：

```text
outputs/vision_ts_routing/expert_predictions/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/
    lightweight_v1__stratified_50k/
      predictions.parquet
      errors.parquet
      manifest.json
      profiling/
        cell_model_matrix.csv
        oracle_summary.csv
```

3. 做分析报告，重点回答：

- oracle ensemble 是否明显优于 best fixed expert；
- 哪些 cell 中 `seasonal_naive` 更强；
- 哪些 cell 中 `linear_trend` 更强；
- valid/test 上专家排序是否稳定；
- 轻量专家是否有资格保留为正式专家池成员。

4. 可以新增分析脚本，但不要改 Stage 1.4b 的正式专家接入脚本。

建议新增：

```text
tools/quitobench_lightweight_expert_analysis.py
```

5. 写实验日志，例如：

```text
experiment_logs/YYYY-MM-DD_HHMM_stage1_4a_lightweight_expert_analysis.md
```

### 窗口 A 新窗口提示

复制下面内容到新窗口：

```text
请在以下工作区执行窗口 A：

/home/user10/TSF/DATAPrepare/.worktrees/stage1-4a-analysis

当前任务是 Stage 1.4a-expanded / analysis，不要进入 Stage 1.4b，不要实现 router，不要运行视觉 encoder，不要接入神经网络专家。

请先阅读：

Doc/视觉伪图像路由项目交接.md
Doc/视觉伪图像路由双路线实施计划.md
Doc/视觉路由实验路线对比_保守验证与推荐主线.md
docs/superpowers/specs/2026-06-07-stage1-4-expert-cache-framework-design.md
docs/superpowers/plans/2026-06-07-stage1-4a-lightweight-expert-cache.md
experiment_logs/2026-06-07_1151_stage1_4a_lightweight_expert_cache.md
experiment_logs/实验日志总览.md

当前 Stage 1.4a smoke 已完成，提交为：

086b923 feat: add stage 1.4a lightweight expert cache

请基于当前轻量专家缓存脚本，先做 stratified 50k 分层扩展，而不是直接全量 62.7 万。

建议 expert_set_id：

lightweight_v1__stratified_50k

分层口径：

split / subset / official_tsf_cell

目标输出：

- predictions.parquet
- errors.parquet
- manifest.json
- profiling/cell_model_matrix.csv
- profiling/oracle_summary.csv

随后分析：

- oracle ensemble 是否明显优于 best fixed expert；
- 哪些 cell 中 seasonal_naive 更强；
- 哪些 cell 中 linear_trend 更强；
- valid/test 上专家排序是否稳定；
- 轻量专家是否值得保留到正式专家池。

所有新增文档、日志、项目语义相关注释必须用中文。每个实验都要写 experiment_logs，并更新 experiment_logs/实验日志总览.md。

请使用：

conda run -n quito ...
```

## 4. 窗口 B：Stage 1.4b 正式专家接入 smoke

### 目标

接入第一个真正需要训练的正式专家，验证外部框架/模型能否复用当前 prediction/error cache schema。

### 建议任务

1. 先做审计，不要直接开长训练：

- Quito repo 是否已有 DLinear / NLinear / PatchTST 等 runner；
- Time-Series-Library/tslib 是否已经在本地；
- 当前 `quito` 环境里依赖是否足够；
- 哪个专家最容易在 QuitoBench registry 上跑通 smoke。

2. 第一候选建议：

```text
DLinear 或 NLinear
```

理由：

- 训练成本低；
- 代码通常更容易复用；
- 适合作为第一个正式训练型专家 smoke。

3. 输出目录必须和窗口 A 分开，例如：

```text
expert_set_id = dlinear_v1__smoke
```

4. Stage 1.4b 第一版目标不是追求效果，而是确认：

- 专家训练数据口径正确；
- 预测能映射回 `physical_window_id`；
- 输出能复用 `predictions.parquet / errors.parquet / manifest.json`；
- train/valid/test 不混淆；
- 不实现 router。

5. 可以新增计划文档，例如：

```text
docs/superpowers/plans/YYYY-MM-DD-stage1-4b-first-framework-expert.md
```

### 窗口 B 新窗口提示

复制下面内容到新窗口：

```text
请在以下工作区执行窗口 B：

/home/user10/TSF/DATAPrepare/.worktrees/stage1-4b-framework

当前任务是 Stage 1.4b：正式训练型专家接入 smoke。不要执行 Stage 1.4a-expanded 分析，不要实现 router，不要运行视觉 encoder。

请先阅读：

Doc/视觉伪图像路由项目交接.md
Doc/视觉伪图像路由双路线实施计划.md
Doc/视觉路由实验路线对比_保守验证与推荐主线.md
docs/superpowers/specs/2026-06-07-stage1-4-expert-cache-framework-design.md
docs/superpowers/plans/2026-06-07-stage1-4a-lightweight-expert-cache.md
experiment_logs/2026-06-07_1151_stage1_4a_lightweight_expert_cache.md
experiment_logs/实验日志总览.md

当前 Stage 1.4a smoke 已完成，提交为：

086b923 feat: add stage 1.4a lightweight expert cache

Stage 1.4b 的目标是接入第一个正式训练型专家，优先考虑 DLinear 或 NLinear。第一步先审计 Quito repo 和 Time-Series-Library/tslib 的可复用 runner，不要直接长训练。

第一版 smoke 的目标是：

- 训练/推理流程能跑通；
- 预测能映射回 physical_window_id；
- 输出复用 Stage 1.4a 的 predictions/errors/manifest/profiling schema；
- train/valid/test 口径正确；
- 不实现 router。

建议 expert_set_id：

dlinear_v1__smoke

所有新增文档、日志、项目语义相关注释必须用中文。每个实验都要写 experiment_logs，并更新 experiment_logs/实验日志总览.md。

请使用：

conda run -n quito ...
```

## 5. 并行时的提交规则

两个窗口可以并行，但要注意：

1. 不要写同一个输出目录。
2. 不要同时大改同一个脚本。
3. 两个窗口都可能改 `experiment_logs/实验日志总览.md`，后提交的一方如果遇到冲突，只需要保留两边新增行。
4. 每个窗口完成一个明确阶段后单独提交。
5. 合并回主线前先运行：

```bash
conda run -n quito python -m pytest tests -q
```

