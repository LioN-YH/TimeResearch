# Stage 1.1：sample-channel light proxy 预计算

## 1. 实验目的

基于 Stage 1.0 正式 working registry，为每个 sample-channel history window 预计算在线可复现的轻量统计 proxy。

本阶段只做 light proxy 预计算和验证：

- 以 `physical_window_id` 为主键。
- 保留 `sample_set_id` 和 `base_registry_id`。
- 只读取 history 窗口 `[history_start_idx, history_end_idx)`。
- 不读取 future target。
- 不做 full STL。
- 不运行专家模型。
- 不实现 router。

## 2. 实验计划

1. 新增 Stage 1.1 测试，先验证红灯。
2. 新增 `tools/quitobench_sample_channel_light_proxy.py`。
3. 第一版采用精简 light proxy 特征集 `light_v1_compact`，覆盖 scale、趋势、波动、自相关和频域复杂度。
4. 运行单元测试与 Stage 1.0 registry 回归测试。
5. 使用正式 working registry 先做 `--max-rows` smoke，再做 627,430 行正式预计算。
6. 读回输出，检查行数、主键唯一性、特征列缺失值和有限值。
7. 写实验日志并更新总览。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
```

预期失败：

```text
ModuleNotFoundError: No module named 'tools.quitobench_sample_channel_light_proxy'
```

实现后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py tests/test_quitobench_window_registry.py -q
```

结果：

```text
4 passed in 1.29s
16 passed in 0.55s
```

smoke：

```bash
conda run -n quito python tools/quitobench_sample_channel_light_proxy.py \
  --max-rows 2000 \
  --progress-every 1000 \
  --output-format csv
```

结果：

```text
proxy_rows=2000
output=outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_rows_2000
split_window_counts={'test': 75, 'train': 1625, 'valid': 300}
subset_window_counts={'hour': 2000}
```

正式预计算：

```bash
conda run -n quito python tools/quitobench_sample_channel_light_proxy.py \
  --progress-every 50000 \
  --output-format auto
```

结果：

```text
proxy_rows=627430
output=outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
split_window_counts={'test': 144335, 'train': 386220, 'valid': 96875}
subset_window_counts={'hour': 403260, 'min': 224170}
```

读回校验：

```bash
conda run -n quito python -c "<读取 manifest 和 parquet，检查行数、主键唯一性、特征缺失值和有限值>"
```

结果：

```text
rows 627430
manifest_rows 627430
unique_physical True
sample_set_id 1 qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
shape (627430, 32)
missing_feature_cells 0
finite_features True
split_counts {'test': 144335, 'train': 386220, 'valid': 96875}
subset_counts {'hour': 403260, 'min': 224170}
```

全量回归测试：

```bash
conda run -n quito python -m pytest tests -q
```

结果：

```text
32 passed in 1.11s
```

## 4. 输入数据与配置

输入 registry：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/config.yml
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/manifest.json
```

输入原始数据：

```text
data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet
data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet
```

proxy 配置：

```text
feature_set=light_v1_compact
recent_fraction=0.25
fft_eps=1e-12
future_read_policy=history_only
uses_full_stl=False
runs_expert_models=False
implements_router=False
```

特征列：

```text
mean
std
median
iqr
min
max
amplitude
last_value
missing_ratio
slope
recent_std_ratio
acf_lag1
acf_period
spectral_entropy
dominant_frequency_strength
```

## 5. 实验结果

新增代码：

```text
tools/quitobench_sample_channel_light_proxy.py
tests/test_quitobench_sample_channel_light_proxy.py
```

正式输出：

```text
outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/sample_channel_proxy.parquet
outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/manifest.json
```

输出规模：

```text
sample_channel_proxy.parquet: 65M
manifest.json: 3.1K
rows: 627,430
columns: 32
feature_columns: 15
```

关键校验：

- `physical_window_id` 唯一。
- `sample_set_id` 唯一且等于正式 working registry 的 `sample_set_id`。
- 输出行数与 Stage 1.0 manifest 的 `total_windows=627430` 一致。
- split/subset 计数与 Stage 1.0 working registry 一致。
- 15 个 proxy 特征列无缺失值且全为有限数值。
- smoke 输出写入 `__smoke_max_rows_2000` 后缀目录，没有覆盖正式输出目录。

## 6. 问题与观察

- 用户担心 “62.7 万 sample-channel 是否偏多” 是合理的；本阶段先按 working registry 全量预计算成功，输出 parquet 约 65M，当前规模可接受。
- 第一版没有纳入更复杂的 proxy。这样可以避免 light proxy 过强、过重，并保持后续视觉 encoder 的结构学习空间。
- `conda run` 会缓冲 stdout，正式运行时进度日志在进程结束后一次性刷新；不影响输出结果。
- 初次 smoke 暴露直接执行脚本时 `tools` namespace 导入失败。根因是 `python tools/xxx.py` 时 `sys.path[0]` 为 `tools/`，项目根不在导入路径。已在 Stage 1.1 脚本中显式加入项目根目录，pytest 与直接执行均已验证。
- `acf_period` 对 `hour` 使用 period 24，对 `min` 使用 period 144；history_len=192，因此两个 subset 都可计算周期 lag 自相关。

## 7. 结论

Stage 1.1 已完成第一版 sample-channel light proxy 预计算。

当前正式 proxy 表以 `physical_window_id` 为主键，保留 `sample_set_id`，只使用 history 窗口计算 15 个轻量 proxy 特征。该输出可供后续统计 baseline、视觉 router 辅助输入、解释分析和可选 proxy prediction 辅助任务使用。

## 8. 下一步计划

1. 进入 Stage 1.2：伪图像协议 smoke test。
2. 继续不要实现 router。
3. 继续不要运行专家模型。
4. Stage 1.2 应复用同一个 `physical_window_id`，并与本阶段 proxy 输出保持可 join。
