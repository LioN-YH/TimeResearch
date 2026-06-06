# QuitoBench item 级质量审计中断与脚本优化记录

## 1. 实验目的

在通道级全量质量指标过慢后，尝试改用 item 级质量口径，并进一步加入等距降采样，以完成 QuitoBench 数据充分性审计中的 forecastability、seasonality strength、trend strength 和 TSF cell 分布统计。

## 2. 实验计划

1. 修改 `tools/quitobench_sufficiency_audit.py`，默认 `--quality-scope item`。
2. 对每个 item 的 5 个指标列做 z-score 后沿通道求均值，构造 item 级代表序列。
3. 先尝试全长 item 级质量指标。
4. 若仍过慢，加入 `--quality-max-points` 等距降采样参数。
5. 运行 2048 点和 512 点降采样版本，观察是否能完成。

## 3. 执行命令

```bash
conda run -n quito python tools/quitobench_sufficiency_audit.py --max-workers 8 --quality-scope item
pkill -P 1361347
kill 1361347
conda run -n quito python tools/quitobench_sufficiency_audit.py --max-workers 8 --quality-scope item --quality-max-points 2048
pkill -f 'tools/quitobench_sufficiency_audit.py --max-workers 8 --quality-scope item --quality-max-points 2048'
conda run -n quito python tools/quitobench_sufficiency_audit.py --max-workers 8 --quality-scope item --quality-max-points 512
ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -C python | rg 'quitobench_sufficiency|PID'
pkill -f 'tools/quitobench_sufficiency_audit.py --max-workers 8 --quality-scope item --quality-max-points 512'
```

## 4. 输入数据与配置

- 数据：
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 质量口径：item 级代表序列。
- 降采样尝试：
  - 全长。
  - 2048 点。
  - 512 点。
- worker 数：8。

## 5. 实验结果

- 全长 item 级质量指标运行约 5 分钟仍无输出，已终止。
- 2048 点降采样版本运行约 4 分钟仍无输出，已终止。
- 512 点降采样版本在用户中断本轮后仍在后台运行；检查时进程已运行约 1 分 41 秒，输出目录仍为空。
- 已按用户中断意图清理 512 点后台进程。
- 当前输出目录 `outputs/data_audit/` 仍为空。
- 脚本已完成的结构性优化：
  - 新增 `--quality-scope {item,channel}`。
  - 新增 `--quality-max-points`。
  - item 级代表序列使用 z-score 后通道均值。
  - 长度、split 和窗口数量统计仍基于完整原始 parquet，不受降采样影响。

## 6. 问题与观察

- 即使 item 级降采样，`statsmodels` STL 在多进程批量任务下仍可能比预期慢，且当前脚本只有任务完成后才写出 CSV，缺少中间进度文件。
- 对“是否足够支撑路线 1/路线 2”的核心判断，窗口数、item 数、split 覆盖和官方 README 的 TSF regime 均衡声明已经提供强证据。
- 当前未生成正式报告，因此尚未完成 Task 0 的最终交付。

## 7. 结论

本轮正式充分性报告尚未完成。下一步应避免继续把 STL 作为阻塞路径：先生成一个不依赖 STL 的确定性规模/窗口/官方 TSF 均衡审计报告，再把质量指标计算改为带进度缓存的小批量任务或更轻量的近似实现。

## 8. 下一步计划

1. 先输出确定性审计：
   - item 数。
   - 原始长度。
   - train/valid/test 长度。
   - 三组窗口设置下的 item/channel 窗口数。
   - 根据 QuitoBench README 记录官方 8-cell 均衡信息和 few-shot 支撑结论。
2. 再补充轻量质量指标：
   - forecastability 可先用 Welch 频谱熵计算。
   - trend 可用线性回归 R2 或首尾平滑斜率近似。
   - seasonality 可用日周期滞后自相关近似。
3. 或者把 Quito 原生 STL 指标改为分批缓存，每批写出中间 CSV，避免长任务无结果。
