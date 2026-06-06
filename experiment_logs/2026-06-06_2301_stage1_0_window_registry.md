# Stage 1.0：QuitoBench 窗口索引与配置注册表

## 1. 实验目的

建立后续 Stage 1.1 light proxy、Stage 1.2 伪图像协议、Stage 1.4 异构专家预测缓存和 Stage 1.5 continuous gate 的统一 sample-channel 主键。

本阶段只生成窗口索引和配置注册表，不计算 proxy，不生成伪图像，不运行专家模型，不实现 router。

## 2. 实验计划

1. 复用 Stage 0 的 QuitoBench revision parquet 和 Stage 0.6b 官方 codebook。
2. 新增 `tools/quitobench_window_registry.py`。
3. 新增 `tests/test_quitobench_window_registry.py`，先确认红灯，再实现脚本。
4. 使用默认 canonical 配置生成全量窗口注册表：
   - `history_len=192`
   - `pred_len=96`
   - `stride=96`
   - `channels=ind_1..ind_5`
   - `subsets=hour,min`
   - `split_strategy=quito_temporal`
5. 输出 `window_index.csv`、`config.yml`、`manifest.json`。
6. 运行测试和输出完整性检查。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

初次失败符合预期：

```text
ModuleNotFoundError: No module named 'tools.quitobench_window_registry'
```

实现后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

结果：

```text
5 passed in 0.49s
```

Smoke：

```bash
conda run -n quito python tools/quitobench_window_registry.py --max-items-per-subset 2
```

全量：

```bash
conda run -n quito python tools/quitobench_window_registry.py
```

回归测试：

```bash
conda run -n quito python -m pytest \
  tests/test_quitobench_window_registry.py \
  tests/test_quitobench_channel_stl_codebook_validation.py \
  tests/test_quitobench_official_cluster_codebook.py -q
```

结果：

```text
14 passed in 5.63s
```

## 4. 输入数据与配置

输入：

- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`
- `outputs/data_audit/quitobench_official_cluster_codebook.csv`

配置：

```text
history_len=192
pred_len=96
stride=96
channels=ind_1,ind_2,ind_3,ind_4,ind_5
subsets=hour,min
split_strategy=quito_temporal
cutoff=2023-07-28 00:00:00
item_level_split=False
```

`quito_temporal` split 口径：

- 每个 item 内部按 `date_time` 排序。
- cutoff 前为 pre-cutoff 序列。
- pre-cutoff 的 80% 为 train，20% 为 valid。
- cutoff 及之后为 test。
- 每个窗口的 history 和 target 都严格落在同一个 split 内。

## 5. 实验结果

新增文件：

- `tools/quitobench_window_registry.py`
- `tests/test_quitobench_window_registry.py`

全量输出：

```text
outputs/vision_ts_routing/window_registry/cfcd86e70e73/window_index.csv
outputs/vision_ts_routing/window_registry/cfcd86e70e73/config.yml
outputs/vision_ts_routing/window_registry/cfcd86e70e73/manifest.json
```

输出规模：

```text
window_index.csv: 601,630 rows x 28 columns, 144M
config_hash: cfcd86e70e73
window_id_unique: True
```

subset / split 窗口数：

| subset | split | windows |
| --- | --- | ---: |
| hour | train | 312,785 |
| hour | valid | 72,380 |
| hour | test | 7,755 |
| min | train | 73,435 |
| min | valid | 11,595 |
| min | test | 123,680 |

官方 TSF cell 窗口数：

| official_tsf_cell | windows |
| --- | ---: |
| highT_highS_highF | 126,160 |
| highT_highS_lowF | 99,930 |
| highT_lowS_highF | 45,900 |
| highT_lowS_lowF | 44,350 |
| lowT_highS_highF | 84,580 |
| lowT_highS_lowF | 109,990 |
| lowT_lowS_highF | 45,630 |
| lowT_lowS_lowF | 45,090 |

manifest 摘要：

```json
{
  "total_windows": 601630,
  "subset_item_counts": {"hour": 517, "min": 773},
  "subset_window_counts": {"hour": 392920, "min": 208710},
  "split_window_counts": {"train": 386220, "valid": 83975, "test": 131435},
  "unique_items": 1290,
  "unique_channels": ["ind_1", "ind_2", "ind_3", "ind_4", "ind_5"]
}
```

## 6. 问题与观察

- 全量脚本当前没有中间进度输出，约数分钟完成；后续如果生成更多 config，可考虑加入 subset/item 级进度和分批写出。
- 输出 CSV 为 144M，已由 `.gitignore` 排除 `outputs/`，不进入 git。
- `start_idx` 定义为 target 起点，也就是 history 结束后的第一个预测位置。
- 当前只实现 `quito_temporal` split；item-level held-out split 可作为后续扩展。
- full STL 不进入 Stage 1.0，也不进入在线路径。

## 7. 结论

Stage 1.0 已完成。

当前已经有稳定的 sample-channel 窗口主键，可供后续 Stage 1.1、Stage 1.2、Stage 1.4 复用。下一步应基于同一个 `window_id` 做 sample-channel light proxy 预计算。

## 8. 下一步计划

1. 进入 Stage 1.1：sample-channel light proxy 预计算。
2. Stage 1.1 应读取 `window_index.csv`，只使用 history 窗口计算在线可复现的轻量 meta-features。
3. 继续不要实现 router，直到 light proxy、伪图像协议和专家预测缓存接口稳定。

## 9. 后续修正：Quito 官方切窗口径

2026-06-06 23:31 已新增修正日志：

```text
experiment_logs/2026-06-06_2331_stage1_0_window_registry_quito_alignment.md
```

本日志中的默认配置 `stride=96` 和 “history/target 严格落在同一 split 内” 只能视为 coarse registry 口径，不再作为 Quito 官方兼容主口径。

修正后的默认口径为：

```text
sample_stride=1
split_context_policy=quito_overlap
```

含义：

- 样本起点逐点滑动，对齐 Quito `TimeSeriesDataset.__len__` 的 `L - seq_len - forecast_horizon + 1`。
- valid/test 的 target 落在当前 split 内，但 history 允许向前借 `history_len` 个上下文点。
- 旧输出 `outputs/vision_ts_routing/window_registry/cfcd86e70e73/` 保留作对照或降采样消融，不再作为正式 Stage 1.1 主输入。
