# Stage 1.4a-expanded：轻量专家缓存 50k 分层分析

## 1. 实验目的

在 Stage 1.4a smoke 已通过的基础上，先做 50,000 条 sample-channel window 的分层扩展，而不是直接运行 627,430 条正式 working registry 全量缓存。

本实验只评估 `last_value`、`seasonal_naive`、`recent_mean`、`linear_trend` 四个 history-only 轻量专家，目标是判断轻量专家缓存是否有保留价值，以及 oracle ensemble 相对 best fixed expert 是否存在可观空间。

本实验明确不实现 router，不运行视觉 encoder，不接入神经网络专家，不修改 Quito 上游代码，不进入 Stage 1.4b。

## 2. 实验计划

1. 在 `tools/quitobench_lightweight_expert_cache.py` 中补充分层抽样入口。
2. 抽样口径固定为 `split / subset / official_tsf_cell`。
3. 使用 `expert_set_id=lightweight_v1__stratified_50k` 生成专家预测缓存。
4. 检查输出文件、唯一主键、soft oracle 权重和 manifest 边界标志。
5. 分析：
   - oracle ensemble 是否明显优于 best fixed expert；
   - 哪些 cell 中 `seasonal_naive` 更强；
   - 哪些 cell 中 `linear_trend` 更强；
   - valid/test 上专家排序是否稳定；
   - 轻量专家是否值得保留到正式专家池。

## 3. 执行命令

聚焦测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

50k 分层缓存：

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --stratified-rows 50000 \
  --stratify-columns split,subset,official_tsf_cell \
  --expert-set-id lightweight_v1__stratified_50k
```

结果分析：

```bash
conda run -n quito python /tmp/stage14a_analyze.py
```

## 4. 输入数据与配置

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- expert_set_id: `lightweight_v1__stratified_50k`
- stratified_rows: `50000`
- stratify_columns: `split,subset,official_tsf_cell`
- random_seed: `20260607`
- 专家集合：`last_value`、`seasonal_naive`、`recent_mean`、`linear_trend`
- 预测输入策略：仅使用 history
- target 使用策略：仅用于 error 和 oracle profiling

## 5. 实验结果

输出目录：

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__stratified_50k/
```

输出文件：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

运行结果：

```text
[done] windows=50000
[done] prediction_rows=200000
[done] latency_ms_per_window=1.4696
```

缓存校验：

```text
manifest_windows 50000
prediction_rows 200000
error_rows 200000
prediction_unique True
error_unique True
soft_weight_max_abs_error 4.440892098500626e-16
implements_router False
runs_visual_encoder False
runs_neural_experts False
```

分层样本量：

| 维度 | 计数 |
| --- | ---: |
| test | 16,775 |
| train | 17,693 |
| valid | 15,532 |
| hour | 20,828 |
| min | 29,172 |

cell 样本量：

| official_tsf_cell | windows |
| --- | ---: |
| highT_highS_highF | 4,645 |
| highT_highS_lowF | 6,675 |
| highT_lowS_highF | 4,645 |
| highT_lowS_lowF | 6,892 |
| lowT_highS_highF | 9,289 |
| lowT_highS_lowF | 8,566 |
| lowT_lowS_highF | 4,644 |
| lowT_lowS_lowF | 4,644 |

oracle summary：

| 指标 | 值 |
| --- | ---: |
| num_windows | 50,000 |
| num_experts | 4 |
| oracle_mse | 3.655510e+10 |
| best_fixed_expert | seasonal_naive |
| best_fixed_mse | 4.040539e+10 |
| uniform_ensemble_mse_proxy | 4.581651e+11 |
| oracle_gap_vs_best_fixed | 3.850293e+09 |

固定专家全局排序：

| expert_id | mean_mse |
| --- | ---: |
| seasonal_naive | 4.040539e+10 |
| recent_mean | 5.328050e+11 |
| linear_trend | 5.333447e+11 |
| last_value | 7.261053e+11 |

valid/test 排序：

| split | rank1 | rank2 | rank3 | rank4 |
| --- | --- | --- | --- | --- |
| test | seasonal_naive | recent_mean | linear_trend | last_value |
| valid | seasonal_naive | linear_trend | recent_mean | last_value |

cell 级第一名：

| official_tsf_cell | best_expert | mean_mse | oracle_top1_rate |
| --- | --- | ---: | ---: |
| highT_highS_highF | seasonal_naive | 1.449202e+07 | 0.315178 |
| highT_highS_lowF | seasonal_naive | 2.078368e+10 | 0.540225 |
| highT_lowS_highF | seasonal_naive | 3.629380e+09 | 0.665016 |
| highT_lowS_lowF | seasonal_naive | 7.804098e+09 | 0.488537 |
| lowT_highS_highF | seasonal_naive | 1.416059e+11 | 0.649478 |
| lowT_highS_lowF | seasonal_naive | 4.616205e+10 | 0.611487 |
| lowT_lowS_highF | seasonal_naive | 9.766205e+06 | 0.322136 |
| lowT_lowS_lowF | seasonal_naive | 2.152900e+10 | 0.484065 |

## 6. 问题与观察

1. `seasonal_naive` 在 8 个 official TSF cell 中全部是平均 MSE 第一名，因此当前四个轻量专家的 cell-level 多样性不足。
2. `linear_trend` 没有在任何 official TSF cell 上取得整体第一名；仅在更细的 `train / highT_lowS_lowF` 子层中出现过第一。这说明它暂时不适合作为强固定专家，但可作为 oracle top-k、多样性和后续 gate 负样本保留。
3. valid/test 上第一名稳定为 `seasonal_naive`，第四名稳定为 `last_value`；第二、第三名在 `recent_mean` 和 `linear_trend` 之间交换，排序基本稳定但区分度有限。
4. oracle MSE 比 best fixed MSE 低约 `3.850293e+09`，相对 best fixed 约下降 9.5%。这个 gap 存在，但不算“明显强到足以单独证明 router 价值”；更像是说明逐窗口选择仍有空间，但当前轻量专家池太弱、太同质。
5. 分层抽样不是每个 cell 完全等量，原因是 `split/subset/official_tsf_cell` 的可用窗口数本身不完全均衡。当前抽样逻辑按非空组合尽量均衡，并在稀缺组合耗尽后把剩余额度分配给仍有样本的组合。

## 7. 结论

轻量专家值得保留到正式专家池，但角色应定位为：

- schema、缓存、oracle profiling 和 gate 训练管线的低成本基线；
- `seasonal_naive` 作为强统计 baseline；
- `recent_mean`、`linear_trend`、`last_value` 作为多样性和负例参考。

当前结果不支持把这四个轻量专家视为充分异构的正式专家池。若要验证视觉先验驱动连续融合的真实价值，后续仍需要在 Stage 1.4b 或之后接入更强且误差形态更互补的专家，但本窗口任务不进入该阶段。

## 8. 下一步计划

1. 保留 `lightweight_v1__stratified_50k` 作为 Stage 1.4a-expanded 分析缓存。
2. 在进入 Stage 1.4b 前，优先明确神经网络专家的 adapter 输出 schema 是否完全复用当前 `predictions/errors/manifest/profiling` 协议。
3. 若继续扩大 Stage 1.4a，可先做 627,430 条全量轻量缓存，但它的主要价值是稳定统计口径，不太可能改变“`seasonal_naive` 统治当前轻量专家池”的核心结论。
