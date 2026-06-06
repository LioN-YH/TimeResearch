# Stage 1.0 修正：Quito 官方切窗口径对齐

## 1. 实验目的

修正 Stage 1.0 窗口 registry 的样本切分口径，使默认行为对齐 Quito `TimeSeriesDataset`，避免把旧版 `strict_within_split + stride=96` 粗粒度索引误用为官方兼容样本索引。

## 2. 实验计划

1. 重读 Quito `quito/quito/datasets.py` 中 `TimeSeriesDataset.__len__`、`_fetch_sample_idx`、`__getitem__` 和 split 边界处理。
2. 用测试锁定 Quito overlap 口径和 strict split 内口径的差异。
3. 修改 `tools/quitobench_window_registry.py`，默认使用 `sample_stride=1` 和 `split_context_policy=quito_overlap`。
4. 保留 `strict_within_split` 作为 coarse registry / smoke / 降采样消融。
5. 运行测试、smoke 和全量规模估算。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

红灯结果：

```text
4 failed, 3 passed
TypeError: RegistryConfig.__init__() got an unexpected keyword argument 'sample_stride'
```

实现后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

结果：

```text
7 passed in 0.42s
```

官方兼容 smoke：

```bash
conda run -n quito python tools/quitobench_window_registry.py --max-items-per-subset 2
```

结果：

```text
config_hash=968e482b1cb0
windows=203,060
split_window_counts={'train': 133440, 'valid': 32880, 'test': 36740}
subset_window_counts={'hour': 148790, 'min': 54270}
```

全量规模估算：

```bash
conda run -n quito python -c "<按 item 长度和 Quito overlap 公式估算窗口数>"
```

结果：

```text
hour {'train': 29874845, 'valid': 7406025, 'test': 1181345} 38462215
min {'train': 6906755, 'valid': 1634895, 'test': 12433705} 20975355
```

## 4. 输入数据与配置

输入源码：

- `quito/quito/datasets.py`

输入数据：

- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`

修正后的默认配置：

```text
history_len=192
pred_len=96
sample_stride=1
split_strategy=quito_temporal
split_context_policy=quito_overlap
channels=ind_1,ind_2,ind_3,ind_4,ind_5
```

## 5. 实验结果

代码变更：

- `tools/quitobench_window_registry.py`
  - `RegistryConfig.stride` 改为更明确的 `sample_stride`。
  - 新增 `split_context_policy`，默认 `quito_overlap`。
  - CLI 新增 `--sample-stride` 和 `--split-context-policy`。
  - CLI 保留 `--stride` 作为旧参数别名。
- `tests/test_quitobench_window_registry.py`
  - 新增 Quito overlap 边界测试。
  - 新增 valid/test history overlap 但 target 不越界测试。

关键源码依据：

- Quito `__len__` 使用 `L - seq_len - forecast_horizon + 1`，没有显式大步长。
- Quito valid/test 的 `border_s` 分别是 `train_size - seq_len` 和 `train_size + valid_size - seq_len`，因此 evaluation split 会向前借历史上下文。
- `__getitem__` 中 `s_begin=j`、`s_end=s_begin+seq_len`、`y_end=s_end+forecast_horizon`，对应 target 起点逐点滑动。

## 6. 问题与观察

- 旧 Stage 1.0 输出 `cfcd86e70e73` 是 `strict_within_split + stride=96` 口径，只有 601,630 条窗口。它可以作为 coarse registry 或降采样消融，但不再作为 Quito 官方兼容主输入。
- 修正后官方兼容全量估算约 59,437,570 条 sample-channel 窗口，不适合直接写单个 CSV。
- `--max-items-per-subset 2` smoke 已经达到 203,060 条，说明 Stage 1.1 需要先定分片/抽样策略。

## 7. 结论

Stage 1.0 的默认脚本口径已修正为 Quito 官方兼容切窗逻辑。后续 Stage 1.1 不应直接基于旧 `cfcd86e70e73` 全量 CSV 做主实验；应先选择：

- 官方兼容分片 Parquet；
- 官方兼容候选窗口抽样；
- 或明确标记为 coarse 消融的 strict 大步长 registry。

## 8. 下一步计划

1. 在 Stage 1.1 开始前确定 registry 生成/读取策略。
2. 如果优先 smoke，可使用 `968e482b1cb0` 的小样本官方兼容 registry。
3. 如果做正式 proxy 预计算，建议先实现 subset/split 分片写出，避免单 CSV 文件过大。
