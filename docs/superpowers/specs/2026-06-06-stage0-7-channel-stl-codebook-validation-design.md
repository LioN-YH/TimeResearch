# Stage 0.7 通道级全长 STL 官方 codebook 验证设计

## 目标

Stage 0.7 用论文 multivariate TSF 口径复核 Stage 0.6b 官方 codebook：对 QuitoBench benchmark 的每个 `(subset, item_id, ind_k)` 运行 full-length `quito.utils.dataset_quality.evaluate_series`，保留通道级结果，再对同一 item 的 5 个 channel 的 `trend_strength`、`seasonality_strength`、`forecastability` 求均值，使用论文固定阈值 `tau=0.4` 得到 `paper_like_tsf_cell`，并与官方 cluster codebook 映射出的 cell 比较。

本阶段不重新定义官方标签，不替代 Stage 0.6b codebook，不实现 router。

## 输入与输出

输入只使用 QuitoBench benchmark：

- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`
- `outputs/data_audit/quitobench_official_cluster_codebook.csv`

输出：

- `outputs/data_audit/quitobench_channel_quality_stl_full.csv`
- `outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv`
- `outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv`
- `outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md`

实验日志：

- `experiment_logs/2026-06-06_1950_stage0_7_channel_stl_codebook_validation.md`
- 同步更新 `experiment_logs/实验日志总览.md`

## 架构

新增单脚本 `tools/quitobench_channel_stl_codebook_validation.py`，职责包括任务生成、通道级质量计算、resume 写 CSV、item 聚合、官方 codebook 比较、报告输出。沿用现有 `tools/` 风格，不修改 `quito/` 官方代码。

新增测试 `tests/test_quitobench_channel_stl_codebook_validation.py`，覆盖可独立测试的数据逻辑：resume key、channel mean 聚合、`tau=0.4` 二值化、官方 cell 比较和 confusion matrix。

## 数据处理规则

- 每个 item 有 5 个 `ind_*` channel。
- `hour` period 使用 24，`min` period 使用 144；若序列过短，period 截断为 `min(period, max(2, len(series)//2))`。
- `evaluate_series` 参数使用 `compute_adf=False`、`compute_hurst=True`，与 Stage 0.1 兼容。
- 通道级 CSV 每批写入，按 `(subset, item_id, channel)` 去重并支持 resume。
- item 聚合只对 `trend_strength`、`seasonality_strength`、`forecastability` 三项求 channel mean；同时保留 `channel_count` 和 `complete_channel_count`。
- 论文二值化规则为 `metric > 0.4` 是 high，`metric <= 0.4` 是 low。

## 验证指标

- item exact match：`paper_like_tsf_cell == official_tsf_cell`
- 逐维 match：`trend_match`、`seasonality_match`、`forecastability_match`
- cluster-level summary：每个官方 cluster 的 exact/dim match 率、均值指标、paper-like cell 众数
- confusion matrix：官方 cell x paper-like cell 的 item 计数
- cluster 24 专项解释：报告其官方 cell、paper-like 众数、exact match 率、三项均值分布，以及与 Stage 0.6 冲突的原因

## 错误处理与恢复

- 缺少输入文件时报错并停止。
- 已有通道级中间 CSV 时默认 resume，跳过已完成 `(subset,item_id,channel)`。
- 每批写 CSV 并打印进度，便于长实验中断后续跑。
- 运行失败也必须保留日志中的命令、失败点和下一步排查方向。
