# Stage 1.4c：训练预算与归一化校准设计

## 1. 设计目的

Stage 1.4b 已经证明 DLinear、PatchTST、TSMixer 可以通过统一 runner 生成与 `physical_window_id` 对齐的 expert cache，并能和 `seasonal_naive` 在同一窗口集合上比较 oracle gap。

但 Stage 1.4b 的 5k stratified smoke 训练样本过少，train split 只有约 1,695 个窗口。这个规模主要适合验证接口、缓存 schema 和互补性统计，不足以判断训练型专家的真实能力。因此 Stage 1.4c 的目标是：

> 在不实现 router/gate、不运行视觉 encoder 的前提下，用更合理的样本量、训练轮数和归一化口径校准现有训练型专家，判断 Stage 1.4b 中 neural experts 弱于 `seasonal_naive` 是否主要来自训练预算不足或尺度处理不当。

本阶段不追求最终最优模型，也不固化最终专家池。它只回答一个前置问题：

> 当前候选训练型专家是否值得进入更昂贵的 OOF cache 和 Stage 1.5 gate 设计？

## 2. 背景判断

Stage 1.4b 的四专家结果显示：

- `seasonal_naive` 是 5k smoke 上的 best fixed expert；
- PatchTST 有约 17.70% 的 oracle top1 rate；
- DLinear 有约 7.86% 的 oracle top1 rate；
- TSMixer 有约 3.40% 的 oracle top1 rate；
- 加入 TSMixer 后 oracle MSE 只略有下降。

这些结果不能直接解释为“训练型专家无效”，原因包括：

1. 训练数据量偏小。5k stratified sample 中 train 仅约 1,695 个窗口，且分散到 `subset/split/official_tsf_cell` 后每个结构区域样本更少。
2. 训练轮数偏低。Stage 1.4b smoke 使用 `epochs=1`，更像接口验证而非能力评估。
3. 归一化口径尚未校准。QuitoBench 中不同 item、通道和 subset 的尺度差异可能让 neural experts 的 MSE 被少数大尺度窗口主导。
4. 当前 comparison 使用同一 5k sample，不足以稳定估计 cell-level 互补性。

因此 Stage 1.4c 应先扩大样本和训练预算，再决定是否新增更多模型或进入 OOF。

## 3. 范围

### 3.1 本阶段要做

- 复用 `tools/quitobench_framework_expert_cache.py`。
- 继续使用 `DLinear / PatchTST / TSMixer` 三个已接入训练型专家。
- 使用同一个更大的 stratified sample，建议第一版为 50k 窗口。
- 固定 `seasonal_naive` full cache 作为 baseline。
- 对每个训练型专家做有限训练预算 sweep。
- 比较不同预算下的整体 MSE、split-level MSE、cell-level MSE、oracle top1 rate 和 oracle gap。
- 记录每个 run 的训练耗时、推理耗时、设备、样本量、epoch、batch size 和归一化策略。
- 生成 Stage 1.4c 实验日志并更新全局日志总览。

### 3.2 本阶段不做

- 不实现 router/gate。
- 不训练或运行视觉 encoder。
- 不做 OOF cache。
- 不新增大量模型。
- 不修改 Quito 上游代码。
- 不把本阶段结果当作最终专家池结论。
- 不用 official TSF cell 训练路线 1 的 gate；cell 仍只用于 profiling。

## 4. 推荐实验矩阵

为了控制成本，第一版不做完整网格搜索，而采用窄矩阵：

| 维度 | 第一版取值 |
| --- | --- |
| sample size | `stratified-rows=50000` |
| models | `dlinear`, `patchtst`, `tsmixer` |
| epochs | `1`, `5`, `20` |
| batch size | `128` 或沿用 Stage 1.4b 默认 |
| device | `cuda` |
| seed | 沿用当前 runner 默认 seed |
| normalization | 先使用 runner 当前口径；如代码已有可切换归一化，再加入 `none` 与 `window_zscore` 对照 |

第一版最小可执行矩阵为 9 个训练 run：

```text
dlinear  x epochs 1/5/20
patchtst x epochs 1/5/20
tsmixer  x epochs 1/5/20
```

如果 50k x 20 epoch 成本过高，可以先执行：

```text
sample size = 50k
epochs = 1 / 5
models = DLinear / PatchTST / TSMixer
```

随后根据 5 epoch 的趋势决定是否补 20 epoch。

## 5. 输出命名

所有输出继续落在：

```text
outputs/vision_ts_routing/expert_predictions/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/
```

建议 `expert_set_id` 使用可读且可比较的命名：

```text
dlinear_v1__stratified_50k_cuda_e1
dlinear_v1__stratified_50k_cuda_e5
dlinear_v1__stratified_50k_cuda_e20
patchtst_v1__stratified_50k_cuda_e1
patchtst_v1__stratified_50k_cuda_e5
patchtst_v1__stratified_50k_cuda_e20
tsmixer_v1__stratified_50k_cuda_e1
tsmixer_v1__stratified_50k_cuda_e5
tsmixer_v1__stratified_50k_cuda_e20
```

每个目录保持现有 cache schema：

```text
predictions.parquet
errors.parquet
manifest.json
profiling/cell_model_matrix.csv
profiling/oracle_summary.csv
```

comparison 输出建议落在：

```text
outputs/vision_ts_routing/expert_comparisons/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/
    budget_calibration_50k__seasonal_naive_dlinear_patchtst_tsmixer/
```

如果不同 epoch 需要独立比较，则用：

```text
budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer
budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer
budget_calibration_50k_e20__seasonal_naive_dlinear_patchtst_tsmixer
```

## 6. 指标

Stage 1.4c 至少报告以下指标：

- `num_common_windows`
- `best_fixed_expert`
- `best_fixed_mse`
- `oracle_mse`
- `oracle_gap_vs_best_fixed`
- 每个 expert 的 `mse`
- 每个 expert 的 `oracle_top1_rate`
- 按 `split` 汇总的 oracle gap
- 按 `official_tsf_cell` 汇总的 best fixed expert、oracle gap 和 expert top1 rate
- 训练耗时与推理耗时
- NaN / inf / 缺失预测比例

解释重点不是单个模型平均 MSE，而是：

1. epoch 增加后 neural experts 是否显著改善；
2. neural experts 是否在某些 cell 或 split 上稳定胜出；
3. oracle gap 是否随训练预算增加而扩大；
4. neural experts 的贡献是否足以支撑后续 OOF 训练成本。

## 7. 与 OOF cache 的关系

OOF cache 暂不在 Stage 1.4c 实现。

原因是 OOF 会把每个训练型专家的训练成本扩大到约 `K` 倍。如果当前候选专家在 50k 非 OOF 校准中仍然没有清晰互补性，直接做 OOF 会浪费大量计算。

Stage 1.4c 的判断逻辑是：

- 如果 50k + 更高 epoch 后，PatchTST/DLinear/TSMixer 的 oracle top1 和 cell-level 贡献明显上升，则进入 OOF cache 设计。
- 如果 neural experts 仍然整体弱且互补性有限，则先重读 TimeFuse、QuitoBench、TimeRecipe、VisMoE 等工作，重新选择覆盖不同结构模式的专家池。

## 8. 停止条件

本阶段完成条件：

1. 至少完成 50k 样本上的 `epochs=1/5` 三模型校准。
2. 生成对应 expert cache 和 comparison summary。
3. 实验日志明确回答：
   - Stage 1.4b 是否低估了 neural experts；
   - 是否值得补 `epochs=20`；
   - 是否值得进入 OOF cache；
   - 是否应该优先新增或替换专家模型。
4. `experiment_logs/实验日志总览.md` 已登记 Stage 1.4c。

如果 50k 运行耗时或显存不可接受，则降级为 20k，但必须在日志中说明降级原因，并保留同一 stratified 口径。

## 9. 风险与约束

- 50k x 多模型 x 多 epoch 可能需要较长 GPU 时间，应先跑一个模型的 `epochs=1` 做耗时估计。
- 如果 runner 训练只使用 train split，50k stratified 的实际 train 数量仍小于 50k，需要在日志中报告 train/valid/test 分布。
- 如果不同 run 的 stratified sample 不一致，comparison 会变得难解释；必须确保同一 sample selection 或在 comparison 中只取共同 `physical_window_id`。
- 如果归一化策略需要新增代码，必须先写测试，且不改变 Stage 1.4b 历史输出语义。
- 如果 neural experts 在 train 上明显变好但 valid/test 不变好，应优先怀疑过拟合，而不是直接进入 gate。

