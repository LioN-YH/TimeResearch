# Stage 1.4c：训练预算与归一化校准

## 1. 目的

Stage 1.4b 的 5k stratified smoke 中，DLinear、PatchTST、TSMixer 的平均 MSE 均弱于 `seasonal_naive`。本实验扩大到 50k stratified sample，并比较 `epochs=1/5`，判断 Stage 1.4b 是否主要低估了训练型专家。

本实验不实现 router/gate，不运行视觉 encoder，不做 OOF cache，不修改 Quito 上游代码。

## 2. 输入与执行口径

窗口 registry：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
```

baseline cache：

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full/
```

训练型专家 runner：

```text
tools/quitobench_framework_expert_cache.py
```

comparison 脚本：

```text
tools/quitobench_expert_cache_comparison.py
```

采样口径：

```text
--stratified-rows 50000
--stratify-cols split,subset,official_tsf_cell
random_seed=20260607
```

实际 split 分布：

```text
train 18980
valid 14820
test  16200
```

## 3. 训练参数

共享参数：

```text
seq_len=192
pred_len=96
enc_in=1
c_out=1
loss=mse
optimizer=Adam
learning_rate=0.001
weight_decay=0.0
batch_size=128
random_seed=20260607
revin=True
```

当前 runner 未启用：

- validation early stopping
- learning rate scheduler
- gradient clipping
- AMP/mixed precision
- DDP/DataParallel
- OOF
- 单独归一化 sweep 参数

模型参数：

```text
DLinear:
  kernel_size=25
  individual=False

PatchTST:
  patch_len=16
  stride=8
  d_model=128
  d_ff=256
  n_heads=4
  e_layers=2
  dropout=0.05
  fc_dropout=0.05
  head_dropout=0.0

TSMixer:
  num_blocks=2
  d_ff=64
  norm_type=layer
  dropout=0.1
```

## 4. 多 GPU 执行情况

使用独立进程按模型分发：

```text
CUDA_VISIBLE_DEVICES=0 -> DLinear
CUDA_VISIBLE_DEVICES=1 -> PatchTST
CUDA_VISIBLE_DEVICES=2 -> TSMixer
```

注意：manifest 中 `device` 均记录为 `cuda:0`，这是因为 runner 在每个独立进程内只看到当前 `CUDA_VISIBLE_DEVICES` 暴露出的单卡。

执行中观察到 GPU 0/1/2 的 CUDA 进程已创建，但 GPU util 多数时间较低。原因不是任务未提交，而是当前 runner 的 CPU 数据准备、单 worker DataLoader、短训练循环和 parquet 写出占主要耗时。

## 5. 输出

expert cache：

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e1/
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e1/
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e1/
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5/
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e5/
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e5/
```

comparison：

```text
outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer/
outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer/
```

所有训练型 expert cache 均为：

```text
windows=50000
prediction_rows=50000
train_windows=18980
```

comparison 均为：

```text
common_windows=50000
experts=seasonal_naive,dlinear_quito,patchtst_quito,tsmixer_quito
```

## 6. 训练耗时

| expert_set_id | total elapsed s | latency ms/window | train elapsed s | final train loss |
| --- | ---: | ---: | ---: | ---: |
| `dlinear_v1__stratified_50k_cuda_e1` | 89.39 | 1.79 | 6.25 | 1305.24 |
| `patchtst_v1__stratified_50k_cuda_e1` | 94.08 | 1.88 | 8.45 | 1340.45 |
| `tsmixer_v1__stratified_50k_cuda_e1` | 90.80 | 1.82 | 7.03 | 1286.67 |
| `dlinear_v1__stratified_50k_cuda_e5` | 63.32 | 1.27 | 9.34 | 1.06 |
| `patchtst_v1__stratified_50k_cuda_e5` | 75.28 | 1.51 | 21.01 | 52679.48 |
| `tsmixer_v1__stratified_50k_cuda_e5` | 64.15 | 1.28 | 10.89 | 264.51 |

## 7. Epoch 1 结果

整体：

```text
num_common_windows 50000
oracle_mse 27801548159.15754
best_fixed_expert seasonal_naive
best_fixed_mse 31981663588.498466
oracle_gap_vs_best_fixed 4180115429.340927
```

专家均值：

```text
seasonal_naive mse=31981663588.498466 oracle_top1_rate=0.6848
dlinear_quito  mse=144834762115.726   oracle_top1_rate=0.2514
tsmixer_quito  mse=280705791585.57733 oracle_top1_rate=0.0528
patchtst_quito mse=111407163768026.9  oracle_top1_rate=0.0110
```

split 层级 oracle gap：

```text
test  1429249929.2392216
train 6855036446.183594
valid 3761358357.9491463
```

cell 层级中 `highT_highS_lowF` 和 `highT_lowS_lowF` 的 best fixed expert 为 `dlinear_quito`，其余 cell 为 `seasonal_naive`。

## 8. Epoch 5 结果

整体：

```text
num_common_windows 50000
oracle_mse 28019560617.348484
best_fixed_expert seasonal_naive
best_fixed_mse 31981663588.498466
oracle_gap_vs_best_fixed 3962102971.1499825
```

专家均值：

```text
seasonal_naive mse=31981663588.498466  oracle_top1_rate=0.70358
dlinear_quito  mse=119621498740.76817  oracle_top1_rate=0.27784
tsmixer_quito  mse=932898898453.8362   oracle_top1_rate=0.01736
patchtst_quito mse=5985860382476923.0  oracle_top1_rate=0.00122
```

split 层级 oracle gap：

```text
test  1625401731.6780186
train 6460389265.80249
valid 3316832134.9112015
```

cell 层级中只有 `highT_lowS_lowF` 的 best fixed expert 为 `dlinear_quito`，其余 cell 为 `seasonal_naive`。

## 9. 结论

Stage 1.4b 的 5k smoke 确实低估了 DLinear 的互补性：扩大到 50k 后，DLinear 的 oracle top1 rate 从 5k smoke 的约 7.86% 提升到 e1 的 25.14%，e5 进一步到 27.78%。DLinear 在部分 low forecastability cell 上有稳定补充价值。

PatchTST 当前结果明显异常。e1 时 MSE 已远大于其他专家，e5 后进一步发散，final train loss 也从约 1340 升至 52679。这不应解释为 PatchTST 模型族无效，更可能说明当前 Quito PatchTST 参数、RevIN/归一化、学习率或 runner 适配口径存在问题。

TSMixer 在 e1 有少量补充价值，但 e5 后平均 MSE 和 oracle top1 rate 均变差，说明当前配置也不适合直接加大 epoch。

本阶段不建议继续盲跑 e20。更合理的下一步是先做训练稳定性和尺度处理排查：

1. 对 DLinear 保留为候选专家，后续可进入 OOF 方案设计。
2. 对 PatchTST 优先做小规模参数/归一化诊断，而不是直接 e20。
3. 对 TSMixer 检查 `d_ff/norm_type/dropout/ReVIN/lr`，确认 e5 变差是否由过拟合或训练不稳造成。
4. 在新增更多模型前，先重读 TimeFuse、QuitoBench、TimeRecipe、VisMoE 的专家选择逻辑，并优先引入覆盖频域/多尺度/强 baseline 的模型。

## 10. 下一步

建议进入 Stage 1.4d：训练型专家稳定性诊断与模型池重选。

建议范围：

- 不实现 router/gate。
- 不运行视觉 encoder。
- 不做 OOF 训练。
- 对 PatchTST/TSMixer 做 5k 或 20k 小矩阵诊断：
  - learning rate `1e-3 / 3e-4 / 1e-4`
  - RevIN on/off
  - batch size `128 / 256`
  - PatchTST 更小 `d_model/d_ff/e_layers`
  - 训练/验证分开报告，避免只看 train loss
- 同步整理 TimeFuse/TimeRecipe/VisMoE/QuitoBench 中的专家族覆盖建议。

