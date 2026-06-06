# Stage 0.7：QuitoBench 通道级 full-length STL 官方 codebook 验证

## 1. 实验目的

按 QuitoBench 论文 multivariate TSF 口径验证 Stage 0.6b 官方 cluster codebook。

本阶段对每个 `(subset, item_id, ind_k)` 运行 full-length `quito.utils.dataset_quality.evaluate_series`，保留通道级中间结果；再对同一 `(subset,item_id)` 的 5 个 channel 的 `trend_strength`、`seasonality_strength`、`forecastability` 分别求均值，用论文固定阈值 `tau=0.4` 二值化得到 `paper_like_tsf_cell`，并与 `quitobench_official_cluster_codebook.csv` 映射出的官方 cell 比较。

本阶段不重新定义官方标签，不替代 Stage 0.6b codebook，不实现 router。

## 2. 实验计划

1. 新增 Stage 0.7 独立脚本：
   - `tools/quitobench_channel_stl_codebook_validation.py`
2. 新增单元测试：
   - `tests/test_quitobench_channel_stl_codebook_validation.py`
3. 通道级 CSV 分批写入并支持 resume：
   - resume key：`(subset, item_id, channel)`
4. 输出 4 个正式产物：
   - `outputs/data_audit/quitobench_channel_quality_stl_full.csv`
   - `outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv`
   - `outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv`
   - `outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md`
5. 报告 item exact match、T/S/F 逐维 match、cluster-level confusion matrix，并专项解释 cluster 24。

## 3. 执行命令

TDD 红灯：

```bash
conda run -n quito python -m pytest tests/test_quitobench_channel_stl_codebook_validation.py -q
```

初次失败原因：

```text
ModuleNotFoundError: No module named 'tools.quitobench_channel_stl_codebook_validation'
```

实现脚本后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_channel_stl_codebook_validation.py -q
```

结果：

```text
6 passed
```

Smoke 命令：

```bash
conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py --max-items-per-subset 2 --max-workers 1 --batch-size 2 --output-dir outputs/data_audit/stage0_7_smoke
```

Smoke 期间发现报告阶段依赖问题：

```text
ImportError: Missing optional dependency 'tabulate'.
```

根因：`pandas.DataFrame.to_markdown()` 依赖可选包 `tabulate`，当前 `quito` conda 环境未安装。修复方式：改用脚本内无依赖 Markdown 表格生成函数 `markdown_table_from_dataframe()`。

修复后 smoke resume 成功：

```text
[start] total=20, completed=20, remaining=0, workers=1, batch_size=2
[done] wrote outputs/data_audit/stage0_7_smoke/quitobench_channel_quality_stl_full.csv
[done] wrote outputs/data_audit/stage0_7_smoke/quitobench_item_quality_stl_channel_mean.csv
[done] wrote outputs/data_audit/stage0_7_smoke/quitobench_official_codebook_channel_stl_validation.csv
[done] wrote outputs/data_audit/stage0_7_smoke/quitobench_official_codebook_channel_stl_validation_report.md
```

全量命令：

```bash
conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py --max-workers 8 --batch-size 25
```

全量结束后刷新报告命令：

```bash
conda run -n quito python tools/quitobench_channel_stl_codebook_validation.py --max-workers 8 --batch-size 25
```

刷新时 resume 命中全部通道：

```text
[start] total=6450, completed=6450, remaining=0, workers=8, batch_size=25
```

## 4. 输入数据与配置

输入：

- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`
- `outputs/data_audit/quitobench_official_cluster_codebook.csv`

数据口径：

- 数据集：`hq-bench/quitobench` benchmark。
- 明确未使用：`hq-bench/quito-corpus` 预训练 corpus。
- `hour` period = 24。
- `min` period = 144。
- `evaluate_series(compute_adf=False, compute_hurst=True)`。
- `tau=0.4`，且严格使用 `> tau` 为 high，`<= tau` 为 low。
- 官方 codebook 继续使用 Stage 0.6b 结论。

Smoke 输出尺寸：

```text
outputs/data_audit/stage0_7_smoke/quitobench_channel_quality_stl_full.csv (20, 18)
outputs/data_audit/stage0_7_smoke/quitobench_item_quality_stl_channel_mean.csv (4, 9)
outputs/data_audit/stage0_7_smoke/quitobench_official_codebook_channel_stl_validation.csv (4, 20)
```

## 5. 实验结果

生成输出：

- `outputs/data_audit/quitobench_channel_quality_stl_full.csv`
- `outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv`
- `outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv`
- `outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md`

输出规模：

```text
outputs/data_audit/quitobench_channel_quality_stl_full.csv (6450, 18)
outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv (1290, 9)
outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv (1290, 20)
outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md 79 lines
```

全量运行进度摘要：

```text
[start] total=6450, completed=0, remaining=6450, workers=8, batch_size=25
[progress] done=6450/6450, batch_seconds=15.7, avg_new_seconds=0.67
[done] wrote outputs/data_audit/quitobench_channel_quality_stl_full.csv
[done] wrote outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv
[done] wrote outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv
[done] wrote outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md
```

总体匹配结果：

| 指标 | match ratio |
| --- | ---: |
| item exact match | 68.37% |
| trend match | 81.32% |
| seasonality match | 69.30% |
| forecastability match | 100.00% |

Cluster 汇总：

| official cluster | official cell | item 数 | paper-like 众数 | exact | T match | S match | F match |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | highT_highS_highF | 166 | highT_highS_highF | 100.00% | 100.00% | 100.00% | 100.00% |
| 2 | highT_highS_lowF | 136 | highT_highS_lowF | 97.06% | 97.79% | 98.53% | 100.00% |
| 6 | highT_lowS_highF | 170 | lowT_highS_highF | 1.76% | 18.82% | 2.94% | 100.00% |
| 8 | highT_lowS_lowF | 157 | lowT_highS_lowF | 9.55% | 39.49% | 13.38% | 100.00% |
| 18 | lowT_highS_highF | 159 | lowT_highS_highF | 95.60% | 98.74% | 96.86% | 100.00% |
| 20 | lowT_highS_lowF | 166 | lowT_highS_lowF | 97.59% | 100.00% | 97.59% | 100.00% |
| 24 | lowT_lowS_highF | 169 | lowT_lowS_highF | 100.00% | 100.00% | 100.00% | 100.00% |
| 26 | lowT_lowS_lowF | 167 | lowT_lowS_lowF | 49.70% | 98.20% | 49.70% | 100.00% |

Cluster 24 专项结论：

- Stage 0.6b 官方 codebook：`24 -> lowT_lowS_highF`。
- 本阶段 channel-mean STL paper-like 众数：`lowT_lowS_highF`，众数占比 100.00%。
- exact match：100.00%；T/S/F match 均为 100.00%。
- Stage 0.6 中 cluster 24 被 item 代表序列口径解释为 `highT_highS_highF` 的冲突，在通道级 full-length STL + channel mean + `tau=0.4` 口径下已消除。

回归测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_channel_stl_codebook_validation.py tests/test_quitobench_official_cluster_codebook.py tests/test_quitobench_item_stl_quality_audit.py -q
```

结果：

```text
12 passed in 13.20s
```

## 6. 问题与观察

- `arch` 不可用会触发 warning，但本阶段 `compute_adf=False`，不影响核心 T/S/F 指标。
- `tabulate` 不可用曾导致报告生成失败，已改为无额外依赖的 Markdown 表格生成。
- Smoke 中第一批 2 条通道耗时较长，主要包含 parquet 读取、Quito import 和首次 STL 初始化开销；后续 batch 约 6-12 秒。
- 全量并行运行中，25 条通道一批，多数 batch 约 10-20 秒，平均新增通道耗时约 0.67 秒。
- Forecastability 维度与 Stage 0.6b 官方 codebook 完全一致，说明官方 codebook 的第三位 digit 语义在本地 channel-mean STL 口径下得到强验证。
- Trend/seasonality 维度总体一致率分别为 81.32% 和 69.30%，但 cluster 6、8 的 T/S 与官方 cell 差异明显，说明本地 `evaluate_series` full-length channel mean 仍不能完全复现论文原始 regime 构造。
- Cluster 24 是关键冲突案例：Stage 0.6 item 代表序列口径冲突明显，本阶段通道均值口径 100% 匹配官方 `lowT_lowS_highF`，支持“Stage 0.6 冲突主要来自代表序列口径”的解释。

## 7. 结论

Stage 0.7 已完成。

结论：

1. 通道级 full-length STL 中间结果已完整保留，覆盖 1,290 个 item、6,450 条通道序列，可供后续通道独立伪图像设计复用。
2. 论文式 channel-mean + `tau=0.4` 口径与 Stage 0.6b 官方 codebook 的 item exact match 为 68.37%。
3. Forecastability 逐维 100% 匹配，强支持官方 codebook 中 forecastability digit 的方向。
4. Cluster 0、2、18、20、24 高度匹配；cluster 6、8 和 26 存在不同程度 T/S 差异，说明本地 Quito `evaluate_series` 口径仍不是官方 regime 构造的完全复现。
5. Stage 0.6b codebook 不应被替代；本阶段用于验证和解释，不产生新的官方标签。

## 8. 下一步计划

1. 后续路线 2A 使用 Stage 0.6b 官方 codebook 作为官方 TSF cell 映射。
2. 后续通道独立伪图像设计可复用 `quitobench_channel_quality_stl_full.csv` 做通道异质性分析。
3. 仍不要实现 router；下一阶段应进入 `vision_ts_routing/` 实验包搭建与路线 1/路线 2 共享模块设计。
