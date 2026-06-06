# Quito/STL 质量指标耗时基准

## 1. 实验目的

回应对 Stage 0 中放弃 Quito/STL 精确质量指标路径的担心：通过少量代表序列计时，估算如果继续采用 Quito 原生 `evaluate_series` / STL 质量指标，全量 item 级和通道级审计大约需要多久。

## 2. 实验计划

1. 编写小型计时脚本 `tools/benchmark_quito_stl_quality.py`。
2. 对 `hour` / `min` 各取第一个 item。
3. 分别测试：
   - item 级代表序列全长。
   - 单通道全长。
   - item 级代表序列等距降采样 2048 点。
4. 记录每个 case 的 `evaluate_series(..., compute_adf=False)` 耗时。
5. 根据单条耗时估算全量审计成本。

## 3. 执行命令

```bash
conda run -n quito python tools/benchmark_quito_stl_quality.py
```

## 4. 输入数据与配置

- 数据：
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- Quito 质量函数：
  - `quito.utils.dataset_quality.evaluate_series`
- ADF：
  - `compute_adf=False`
- STL period：
  - `hour`: 24
  - `min`: 144
- 代表序列：
  - item 级：5 个指标列 z-score 后沿通道求均值。
  - channel 级：`ind_1`。

## 5. 实验结果

计时结果：

| subset | case | length | period | seconds |
| --- | --- | ---: | ---: | ---: |
| hour | item_mean_z_full | 15,356 | 24 | 6.728 |
| hour | single_channel_full | 15,356 | 24 | 6.843 |
| hour | item_mean_z_2048 | 2,048 | 24 | 0.614 |
| min | item_mean_z_full | 5,904 | 144 | 6.672 |
| min | single_channel_full | 5,904 | 144 | 6.493 |
| min | item_mean_z_2048 | 2,048 | 144 | 2.098 |

粗略估算：

- item 级全长：
  - 1,290 条 item * 约 6.7 秒 / 8 worker ≈ 18 分钟。
  - 考虑 parquet 读取、进程调度和长尾，预计约 20-30 分钟。
- 通道级全长：
  - 6,450 条 item-channel * 约 6.7 秒 / 8 worker ≈ 90 分钟。
  - 考虑长尾和调度，预计约 1.5-2 小时。
- item 级 2048 点降采样：
  - hour 约 0.6 秒/条，min 约 2.1 秒/条。
  - 预计约 4-8 分钟。

## 6. 问题与观察

- 之前在 5-9 分钟没有输出就终止 STL 路径，确实偏保守；当时脚本只有全部任务完成后才写出结果，没有中间进度，因此无法判断是否接近完成。
- 通道级全长 STL 不适合作为快速 Stage 0 默认路径，但不是不可完成任务；预计 1.5-2 小时可完成。
- item 级全长 STL 是可行的，预计 20-30 分钟。
- 2048 点 item 级 STL 降采样也可行，预计 4-8 分钟，但降采样会显著改变 seasonality/trend 数值，不能完全替代全长 STL。

## 7. 结论

用户对“是否过早放弃 STL 精确质量指标”的担心成立。当前 Stage 0 的轻量 proxy 审计足以支持“QuitoBench-only 是否可行”的规模判断，但如果要为路线 2 的 TSF cell 提供更稳健前置准备，建议补做 item 级全长 Quito/STL 精确质量审计。

推荐优先级：

1. 补做 item 级全长 STL 审计，预计 20-30 分钟。
2. 如路线 2 需要通道级 cell，再安排通道级全长 STL 审计，预计 1.5-2 小时。
3. 给 STL 审计脚本加中间进度和分批写 CSV，避免长时间无输出。

## 8. 下一步计划

在新窗口中，Stage 0.5 前可以先补一个 “Stage 0.1：Quito/STL 精确质量指标审计”：

- 输出 `outputs/data_audit/quitobench_item_quality_stl.csv`。
- 输出 `outputs/data_audit/quitobench_stl_cell_distribution.csv`。
- 更新 `quitobench_sufficiency_report.md` 或新增 STL 复核报告。
- 每批写中间文件，避免再次因为无输出而误判。
