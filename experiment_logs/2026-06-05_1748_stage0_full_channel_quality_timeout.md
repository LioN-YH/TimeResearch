# QuitoBench 通道级全量质量审计超时

## 1. 实验目的

尝试对 QuitoBench benchmark 的 1,290 个 item × 5 个通道执行全量 Quito 质量指标计算，用于同时获得通道级 forecastability、seasonality strength、trend strength 和 TSF cell 分布。

## 2. 实验计划

1. 使用 `tools/quitobench_sufficiency_audit.py` 读取 `hour` / `min` parquet。
2. 精确统计 item 级长度和 train/valid/test 窗口数。
3. 对所有 `(item_id, ind_k)` 单变量序列运行 `quito.utils.dataset_quality.evaluate_series`。
4. 输出数据充分性报告和 CSV。

## 3. 执行命令

```bash
conda run -n quito python tools/quitobench_sufficiency_audit.py --max-workers 8
ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -C python | head -40
pkill -P 1337725
kill 1337725
```

## 4. 输入数据与配置

- `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 通道级质量任务数量：`(517 + 773) * 5 = 6,450`
- STL 周期：
  - `hour`: 24
  - `min`: 144
- worker 数：8

## 5. 实验结果

- 进程运行约 9 分钟后仍未输出结果文件。
- `ps` 显示 8 个 worker 持续接近满 CPU，说明主要耗时在 STL 质量指标计算，而不是 parquet 读取或窗口计数。
- 已手动终止该审计进程，退出码为 143。

## 6. 问题与观察

- 对 6,450 条较长序列做全量 STL 分解成本偏高，不适合作为当前数据充分性审计的默认路径。
- 当前核心决策是“只用 QuitoBench 是否足够支撑路线 1 和路线 2”，其中最关键的是 item 数、窗口数、split 覆盖、TSF cell 覆盖和 few-shot 支撑。
- QuitoBench README 已说明 1,290 条 item 序列在 8 个 TSF regime cell 上分层均衡；因此当前可先用 item 级全量质量指标重构 cell，并用 item cell 复制到通道级样本量口径，匹配默认通道独立策略。

## 7. 结论

通道级全量 STL 质量审计暂不作为默认路径。需要优化脚本为 item 级全量质量指标 + 通道级精确窗口统计；逐通道质量指标可以作为后续更细粒度分析单独运行或抽样运行。

## 8. 下一步计划

1. 修改审计脚本，默认使用 item 级质量口径。
2. 对每个 item 先对 5 个指标列做 z-score 后求均值，形成 item 级代表序列，再计算 forecastability、seasonality strength、trend strength。
3. 输出 item 级 TSF cell 分布，并将 item cell 映射到通道独立样本量口径。
4. 重新运行审计并生成正式报告。
