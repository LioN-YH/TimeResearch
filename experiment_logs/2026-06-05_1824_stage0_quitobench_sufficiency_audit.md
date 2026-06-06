# QuitoBench 数据充分性正式审计

## 1. 实验目的

确认只使用 QuitoBench benchmark 是否足够支撑视觉伪图像路由项目的路线 1 和路线 2，并生成 Task 0 要求的审计报告与 CSV 输出。

## 2. 实验计划

1. 不再把 Quito/STL 质量指标作为阻塞路径。
2. 使用完整原始 parquet 精确统计：
   - item 数。
   - 原始长度。
   - train/valid/test 长度。
   - `seq_len=96/192/336, pred_len=96` 的 item/channel 窗口数。
3. 使用轻量、可复现的 proxy 质量指标复核 TSF cell 覆盖：
   - forecastability：FFT 频谱熵。
   - seasonality：日周期滞后自相关。
   - trend：线性趋势 R2。
4. 输出报告和 CSV。
5. 检查输出文件、行数和报告关键结论。

## 3. 执行命令

```bash
conda run -n quito python tools/quitobench_sufficiency_audit.py --max-workers 1 --quality-scope item --quality-method light --quality-max-points 2048
rg -n "daily period|路线 1|路线 2|后续需优先" outputs/data_audit/quitobench_sufficiency_report.md
ls -lh outputs/data_audit
wc -l outputs/data_audit/*.csv outputs/data_audit/*.md
```

## 4. 输入数据与配置

- 数据集：`hq-bench/quitobench`
- 本地数据：
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 明确未使用：`hq-bench/quito-corpus`
- cutoff：`2023-07-28 00:00:00`
- split 规则：
  - cutoff 前为 train+valid。
  - valid = `int(pre_cutoff_len * 0.2)`。
  - train = `pre_cutoff_len - valid`。
  - cutoff 及之后为 test。
- 窗口设置：
  - `seq_len=96, pred_len=96`
  - `seq_len=192, pred_len=96`
  - `seq_len=336, pred_len=96`
- 质量口径：
  - `--quality-scope item`
  - `--quality-method light`
  - `--quality-max-points 2048`
- 默认通道独立策略：
  - 通道样本数和窗口数按 5 个指标列扩展。
  - 通道级 cell 由 item 级 cell 复制得到，用于样本量审计。

## 5. 实验结果

生成输出：

- `outputs/data_audit/quitobench_sufficiency_report.md`
- `outputs/data_audit/quitobench_cell_distribution.csv`
- `outputs/data_audit/quitobench_window_counts.csv`
- `outputs/data_audit/quitobench_item_lengths.csv`
- `outputs/data_audit/quitobench_item_quality.csv`
- `outputs/data_audit/quitobench_channel_quality.csv`

输出行数：

- `quitobench_cell_distribution.csv`：145 行。
- `quitobench_channel_quality.csv`：6,451 行。
- `quitobench_item_lengths.csv`：1,291 行。
- `quitobench_item_quality.csv`：1,291 行。
- `quitobench_window_counts.csv`：11,611 行。
- `quitobench_sufficiency_report.md`：100 行。

核心规模：

- `hour`：7,939,052 rows，517 item，5 个指标列。
- `min`：4,563,792 rows，773 item，5 个指标列。
- 合计：1,290 item，通道独立口径为 6,450 条 item-channel 序列。

窗口数量要点：

- `hour, seq_len=336, pred_len=96`：
  - train item windows：5,900,521。
  - valid item windows：1,307,493。
  - test item windows：62,557。
  - 通道独立 test windows：312,785。
- `min, seq_len=336, pred_len=96`：
  - train item windows：1,270,039。
  - valid item windows：67,251。
  - test item windows：2,227,013。
  - 通道独立 test windows：11,135,065。

TSF cell 复核：

- 官方 README 声明 1,290 条 item 序列在 8 个 TSF regime cell 上分层均衡，约 160 条序列/cell。
- 轻量 proxy cell combined item 口径覆盖 8 个 cell，最小 cell 为 43 个 item。
- 轻量 proxy cell combined 通道口径覆盖 8 个 cell，最小 cell 为 215 个通道样本。
- subset 内 proxy cell 不完全均衡，因此路线 2 后续应优先定位官方 TSF regime 标签；若不可得，应采用全局分位或合并稀疏 cell。

## 6. 问题与观察

- 当前 parquet 没有显式 TSF cell 标签列。
- Quito/STL 精确质量指标在本机全量运行较慢，已改为轻量 proxy 指标完成充分性审计。
- 轻量 proxy 指标用于审计“覆盖与样本量”，不等同于 Quito/STL 精确质量标签。
- 当前审计仍不能回答专家间 oracle gap；该问题需要后续生成多专家逐窗口预测缓存。

## 7. 结论

只使用 QuitoBench 足够支撑路线 1 第一阶段和路线 2 第一阶段：

- 路线 1：窗口数充足，可直接做普通样本级专家融合/路由能力验证。
- 路线 2：官方 README 提供 8 个 TSF regime cell 均衡依据；轻量 proxy 复核显示 combined 口径覆盖 8 个 cell，并支持 few-shot 分析，但需要后续定位官方 TSF 标签或制定 proxy cell 合并策略。

TimeFuse 数据集暂不作为第一阶段必需项，仅保留为外部泛化检查。

## 8. 下一步计划

1. 优先定位 QuitoBench 官方 TSF regime 标签是否有单独文件或可从项目页/论文补充材料获得。
2. 若官方标签不可得，为路线 2 固化 proxy cell 构造规则，并合并极稀疏 cell。
3. 进入下一阶段前，准备专家 profiling 所需的逐窗口预测缓存方案，但不要在当前阶段实现 router。
