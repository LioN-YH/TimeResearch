# Stage 0.1：QuitoBench item 级全长 STL 精确质量审计

## 1. 实验目的

使用 Quito 原生质量函数 `quito.utils.dataset_quality.evaluate_series`，补做 QuitoBench benchmark 的 item 级全长 STL 精确质量指标审计，并基于精确指标重构 TSF cell 分布，再与 Stage 0 的 light proxy 结果对比。

本实验只做质量指标和 TSF cell 审计，不做伪图像、不做 router、不使用 `hq-bench/quito-corpus` 预训练 corpus。

## 2. 实验计划

1. 新增 Stage 0.1 专用脚本 `tools/quitobench_item_stl_quality_audit.py`。
2. 每个 item 的 5 个指标列先逐通道 z-score，再沿通道取均值，构造 item 级代表序列。
3. 对 1,290 个 QuitoBench item 使用全长序列调用 `evaluate_series(period=24/144, compute_adf=False)`。
4. 使用 8 worker 并行，每 25 条结果写一次中间 CSV，支持断点续跑。
5. 基于 STL 精确质量指标的全局中位数阈值构造 2x2x2 TSF cell。
6. 生成报告，并与 Stage 0 `quitobench_item_quality.csv` light proxy cell 对比。
7. 运行轻量单元测试和输出完整性检查。

## 3. 执行命令

```bash
conda run -n quito python -m pytest tests/test_quitobench_item_stl_quality_audit.py -q
conda run -n quito python tools/quitobench_item_stl_quality_audit.py --max-workers 8 --batch-size 25
wc -l outputs/data_audit/quitobench_item_quality_stl.csv outputs/data_audit/quitobench_stl_cell_distribution.csv outputs/data_audit/quitobench_stl_quality_report.md
python - <<'PY'
import pandas as pd
q = pd.read_csv('outputs/data_audit/quitobench_item_quality_stl.csv')
c = pd.read_csv('outputs/data_audit/quitobench_stl_cell_distribution.csv')
print(q.shape)
print(q['subset'].value_counts())
print(q.duplicated(['subset','item_id']).sum())
print(q[['forecastability','seasonality_strength','trend_strength']].isna().sum())
print(c[(c['subset']=='combined') & (c['seq_len']==96)][['tsf_cell','unit_count']].sort_values('unit_count'))
PY
```

## 4. 输入数据与配置

- 数据集：`hq-bench/quitobench` benchmark。
- 使用 config：`hour`、`min`。
- 本地 parquet：
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 明确未使用：`hq-bench/quito-corpus` 预训练 corpus。
- item 数：
  - `hour`: 517
  - `min`: 773
  - 合计：1,290
- daily period：
  - `hour`: 24
  - `min`: 144
- Quito 质量函数：
  - `quito.utils.dataset_quality.evaluate_series`
  - `compute_adf=False`
  - `compute_hurst=True`
- 并行与落盘：
  - `--max-workers 8`
  - `--batch-size 25`
  - 中间 CSV：`outputs/data_audit/quitobench_item_quality_stl.csv`

## 5. 实验结果

生成输出：

- `outputs/data_audit/quitobench_item_quality_stl.csv`
- `outputs/data_audit/quitobench_stl_cell_distribution.csv`
- `outputs/data_audit/quitobench_stl_quality_report.md`

输出行数：

- `quitobench_item_quality_stl.csv`：1,291 行，含表头；1,290 条 item 结果。
- `quitobench_stl_cell_distribution.csv`：73 行，含表头。
- `quitobench_stl_quality_report.md`：78 行。

完整性检查：

- `quitobench_item_quality_stl.csv` shape：`(1290, 21)`。
- subset 分布：`hour=517`，`min=773`。
- `(subset, item_id)` 重复数：0。
- `forecastability`、`seasonality_strength`、`trend_strength` 缺失数均为 0。
- 单元测试：`3 passed`。

核心 STL 指标分布：

| subset | metric | min | p25 | p50 | p75 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hour | forecastability | 0.0490 | 0.3510 | 0.4575 | 0.5249 | 0.6864 |
| hour | seasonality_strength | 0.0101 | 0.5792 | 0.7950 | 0.8659 | 0.9783 |
| hour | trend_strength | 0.0569 | 0.2874 | 0.4880 | 0.7159 | 0.9984 |
| min | forecastability | 0.0526 | 0.3186 | 0.4839 | 0.6527 | 0.7943 |
| min | seasonality_strength | 0.0000 | 0.6478 | 0.8636 | 0.9504 | 0.9935 |
| min | trend_strength | 0.0000 | 0.1831 | 0.3460 | 0.5764 | 0.9819 |

全局中位数阈值：

- `trend_threshold`: 0.393544
- `seasonality_threshold`: 0.828502
- `forecastability_threshold`: 0.471063

combined item 口径 `seq_len=96` 的 STL cell 分布：

| tsf_cell | item 数 |
| --- | ---: |
| highT_highS_lowF | 22 |
| lowT_lowS_highF | 56 |
| highT_lowS_highF | 59 |
| lowT_highS_lowF | 93 |
| lowT_highS_highF | 150 |
| highT_lowS_lowF | 184 |
| lowT_lowS_lowF | 346 |
| highT_highS_highF | 380 |

与 Stage 0 light proxy 对比：

- 匹配 item 数：1,290。
- STL cell 与 proxy cell 完全一致：581，占 45.04%。
- Spearman 相关：
  - forecastability：0.9053
  - seasonality_strength：0.4089
  - trend_strength：0.6661

## 6. 问题与观察

- `conda run` 在长实验期间缓冲了脚本 stdout，进度行到进程结束后才集中回放；但中间 CSV 正常每批写入，可通过 `wc -l` 观察进度。
- 8 worker 持续满载，单条全长 STL 耗时中位数约 6.6 秒，和 Stage 0 基准一致。
- `arch` 仍不可用，因此 ADF 不计算；本实验显式使用 `compute_adf=False`，不影响 forecastability、seasonality、trend 三项核心质量指标。
- STL 精确 cell 与 light proxy cell 只有 45.04% 完全一致，说明 Stage 0 proxy 只能作为规模充分性复核，不能直接替代最终 cell 标签。
- STL combined item 口径覆盖 8 个 cell，但极小 cell 只有 22 个 item，不支持所有 cell 的 50-shot；这与 Stage 0 proxy 的稀疏 cell 观察一致。

## 7. 结论

Stage 0.1 已完成 item 级全长 Quito/STL 精确质量指标审计。结果覆盖全部 1,290 个 QuitoBench benchmark item，生成了 STL 精确质量 CSV、STL cell 分布 CSV 和中文报告。

STL 指标应作为 Stage 0.5 中“若找不到官方 TSF regime/cell 标签，则固化最终 cell 构造规则”的主要候选依据。当前仍不建议直接进入 router；下一步应先定位官方 TSF regime/cell 标签是否存在。

## 8. 下一步计划

1. 进入 Stage 0.5：定位 QuitoBench 官方 TSF regime/cell 标签是否存在。
2. 如果找不到官方标签，基于 STL 精确指标和 Stage 0 light proxy 对比固化最终 cell 构造规则。
3. 如路线 2 后续需要通道级精确 cell，再单独安排通道级全长 STL 长实验。
4. 当前仍不要实现 router。
