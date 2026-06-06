# Stage 1.0 修正：窗口 ID schema 稳定化

## 1. 实验目的

在 Stage 1.1 开始前稳定窗口索引主键，避免同一个物理窗口因为来自不同采样策略而生成不同 `window_id`，导致 proxy、伪图像和专家预测缓存后续返工。

## 2. 实验计划

1. 新增测试：同一个物理窗口在 `sample_stride=1` 和 `sample_stride=2` 中应共享同一个 `physical_window_id`。
2. 新增测试：不同采样策略应产生不同 `sample_set_id`。
3. 修改 `tools/quitobench_window_registry.py`，分离三层 ID：
   - `physical_window_id`
   - `base_registry_id`
   - `sample_set_id`
4. 保留 `window_id` 作为兼容别名，当前等于 `physical_window_id`。
5. 运行单测和 smoke。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

红灯结果：

```text
ImportError: cannot import name 'build_sample_set_id'
```

实现后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
```

结果：

```text
9 passed in 0.51s
```

Smoke：

```bash
conda run -n quito python tools/quitobench_window_registry.py --max-items-per-subset 1 --sample-stride 96
```

结果：

```text
sample_set_id=qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
windows=1070
split_window_counts={'train': 700, 'valid': 175, 'test': 195}
subset_window_counts={'hour': 780, 'min': 290}
```

字段检查：

```bash
conda run -n quito python -c "读取 smoke CSV 并打印 physical_window_id/window_id/base_registry_id/sample_set_id"
```

结果确认：

```text
physical_window_id == window_id
base_registry_id = qb_h192_p96_quito_overlap_8478f330
sample_set_id = qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
```

## 4. 输入数据与配置

输入：

- `tools/quitobench_window_registry.py`
- `tests/test_quitobench_window_registry.py`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`

Smoke 配置：

```text
history_len=192
pred_len=96
sample_stride=96
split_context_policy=quito_overlap
max_items_per_subset=1
```

## 5. 实验结果

代码变更：

- 新增 `build_base_registry_id(config)`。
- 新增 `build_sample_set_id(config)`。
- `physical_window_id` 不包含 `sample_stride`，只描述真实 sample-channel 窗口身份。
- `sample_set_id` 包含 `sample_stride` 和 channel/subset 等采样集合配置。
- `window_id` 暂时保留为兼容列，值等于 `physical_window_id`。
- 输出目录改为 `outputs/vision_ts_routing/window_registry/<sample_set_id>/`。

## 6. 问题与观察

- 这是减少后续返工的关键 schema 修正。后续即使增加 stratified sample set，只要物理窗口相同，就可以复用已有 proxy、伪图像和专家预测缓存。
- `config_hash` 仍保留，但不再建议作为主要语义 ID。
- 当前 smoke 输出不是正式 working registry，只用于验证 schema。

## 7. 结论

Stage 1.0 的 ID schema 已稳定为三层：

```text
physical_window_id -> 缓存主键
base_registry_id   -> 合法窗口母体
sample_set_id      -> 具体采样集合
```

Stage 1.1 可以基于 `physical_window_id` 设计 proxy 表，基于 `sample_set_id` 管理实验数据子集。

## 8. 下一步计划

1. 正式生成 `quito_overlap + sample_stride=96` working registry。
2. 进入 Stage 1.1：对 working registry 计算 sample-channel light proxy。
3. 后续若需要结构均衡，新增 stratified `sample_set_id`，不覆盖已有 registry。

## 9. 追加修正：默认 working stride

同一阶段继续修正默认值：`RegistryConfig.sample_stride` 和 CLI `--sample-stride` 默认值已从 `1` 改为 `96`。

原因：`sample_stride=1` 对应约 5,943.8 万 sample-channel 全量母体，用户直接运行脚本时容易误生成超大 registry；当前第一版 Stage 1.1 working set 推荐 `quito_overlap + sample_stride=96`，约 62.7 万 sample-channel。若后续确实需要全量母体，应显式传入 `--sample-stride 1` 并使用分片/估算流程。

补充验证：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py tests/test_quitobench_channel_stl_codebook_validation.py tests/test_quitobench_official_cluster_codebook.py -q
```

结果：

```text
10 passed in 0.42s
19 passed in 0.53s
```

## 10. 正式 working registry 生成

继续执行默认命令，生成 Stage 1.1 推荐输入 registry：

```bash
conda run -n quito python tools/quitobench_window_registry.py
```

结果：

```text
sample_set_id=qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
windows=627430
split_window_counts={'train': 386220, 'valid': 96875, 'test': 144335}
subset_window_counts={'hour': 403260, 'min': 224170}
```

输出：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/config.yml
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/manifest.json
```

文件大小：

```text
window_index.csv: 221M
config.yml: 325B
manifest.json: 1.2K
```

该 registry 是 Stage 1.1 的当前正式 working input。

## 11. Review 修正：smoke/full 输出隔离

审查发现问题：`--max-items-per-subset` 不进入 `sample_set_id`，导致 smoke registry 和正式 full registry 会写入同一个目录，后运行者覆盖先运行者。

修正：

- `write_registry_outputs()` 新增 `run_scope` 和 `max_items_per_subset`。
- full registry 仍写入 `<sample_set_id>/`。
- smoke registry 写入 `<sample_set_id>__smoke_max_items_N/`。
- manifest 新增 `run_scope`、`max_items_per_subset`、`output_dir_name`。
- CLI 根据 `--max-items-per-subset` 自动设置 `run_scope=smoke/full`。

验证命令：

```bash
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py -q
conda run -n quito python tools/quitobench_window_registry.py --max-items-per-subset 1 --sample-stride 96
conda run -n quito python tools/quitobench_window_registry.py
conda run -n quito python -m pytest tests/test_quitobench_window_registry.py tests/test_quitobench_channel_stl_codebook_validation.py tests/test_quitobench_official_cluster_codebook.py -q
```

结果：

```text
11 passed in 0.44s
smoke output=outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1
full output=outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
20 passed in 0.52s
```

正式 full manifest 已确认：

```text
total_windows=627430
run_scope=full
max_items_per_subset=None
output_dir_name=qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
```
