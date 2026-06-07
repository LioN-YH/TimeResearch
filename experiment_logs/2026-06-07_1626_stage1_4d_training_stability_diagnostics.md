# Stage 1.4d：训练型专家稳定性诊断

## 1. 目的

Stage 1.4c 显示：

- DLinear 在 50k/e1/e5 中有稳定互补性；
- PatchTST 在当前配置下 e5 明显发散；
- TSMixer e5 变差；
- 不适合直接盲跑 e20。

本阶段用 20k stratified sample 做小矩阵诊断，判断 PatchTST/TSMixer 的问题是否来自学习率、模型容量或 RevIN 口径。

本实验不实现 router/gate，不运行视觉 encoder，不做 OOF cache，不修改 Quito 上游代码。

## 2. 代码改动

为支持诊断，给 `tools/quitobench_framework_expert_cache.py` 增加了显式训练开关：

```text
--weight-decay
--dropout
--fc-dropout
--head-dropout
--revin
--no-revin
```

测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py -q
```

结果：

```text
14 passed
```

提交：

```text
8c31eb6 feat: expose expert runner diagnostic training flags
```

## 3. 采样与 baseline

采样口径：

```text
--stratified-rows 20000
--stratify-cols split,subset,official_tsf_cell
random_seed=20260607
```

实际 split 分布：

```text
train 7189
valid 5916
test  6895
```

为避免 20k 诊断 cache 与 50k DLinear cache 的窗口集合不完全重叠，本阶段额外补跑同一 20k sample 的 DLinear e5 参考：

```text
dlinear_v1__stratified_20k_cuda_e5
```

所有 comparison 均以 20k common windows 完成。

## 4. 诊断矩阵

PatchTST：

```text
patchtst_v1__stratified_20k_cuda_e5_lr3e4
  lr=0.0003, default capacity, RevIN on

patchtst_v1__stratified_20k_cuda_e5_small_lr3e4
  lr=0.0003, d_model=64, d_ff=128, e_layers=1, RevIN on

patchtst_v1__stratified_20k_cuda_e5_lr3e4_no_revin
  lr=0.0003, default capacity, RevIN off
```

TSMixer：

```text
tsmixer_v1__stratified_20k_cuda_e5_lr3e4
  lr=0.0003, num_blocks=2, d_ff=64, norm_type=layer, RevIN on

tsmixer_v1__stratified_20k_cuda_e5_lr3e4_no_revin
  lr=0.0003, num_blocks=2, d_ff=64, norm_type=layer, RevIN off
```

DLinear reference：

```text
dlinear_v1__stratified_20k_cuda_e5
  lr=0.001, kernel_size=25, RevIN on
```

## 5. 训练统计

| expert_set_id | elapsed s | train_windows | epochs | lr | RevIN | final_train_loss | train_elapsed s |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dlinear_v1__stratified_20k_cuda_e5` | 31.01 | 7189 | 5 | 0.001 | True | 0.52 | 3.06 |
| `patchtst_v1__stratified_20k_cuda_e5_lr3e4` | 45.26 | 7189 | 5 | 0.0003 | True | 24986.17 | 7.95 |
| `patchtst_v1__stratified_20k_cuda_e5_small_lr3e4` | 43.88 | 7189 | 5 | 0.0003 | True | 41.51 | 6.04 |
| `patchtst_v1__stratified_20k_cuda_e5_lr3e4_no_revin` | 45.66 | 7189 | 5 | 0.0003 | False | 3251629568.00 | 8.06 |
| `tsmixer_v1__stratified_20k_cuda_e5_lr3e4` | 36.05 | 7189 | 5 | 0.0003 | True | 1.50 | 4.42 |
| `tsmixer_v1__stratified_20k_cuda_e5_lr3e4_no_revin` | 35.91 | 7189 | 5 | 0.0003 | False | 15462174720.00 | 4.43 |

## 6. Comparison 结果

### 6.1 PatchTST lr=3e-4

```text
num_common_windows 20000
best_fixed_expert seasonal_naive
best_fixed_mse 26413584764.11275
oracle_mse 22573094834.04426
oracle_gap_vs_best_fixed 3840489930.068493
```

```text
seasonal_naive oracle_top1_rate=0.69955 mse=2.641358e10
dlinear_quito  oracle_top1_rate=0.29775 mse=1.322765e11
patchtst_quito oracle_top1_rate=0.00270 mse=1.336924e15
```

### 6.2 PatchTST small lr=3e-4

```text
num_common_windows 20000
best_fixed_expert seasonal_naive
best_fixed_mse 26413584764.11275
oracle_mse 22556029230.12964
oracle_gap_vs_best_fixed 3857555533.983112
```

```text
seasonal_naive oracle_top1_rate=0.69735 mse=2.641358e10
dlinear_quito  oracle_top1_rate=0.29020 mse=1.322765e11
patchtst_quito oracle_top1_rate=0.01245 mse=1.604489e14
```

### 6.3 PatchTST lr=3e-4 no RevIN

```text
num_common_windows 20000
best_fixed_expert seasonal_naive
best_fixed_mse 26413584764.11275
oracle_mse 22300283129.03217
oracle_gap_vs_best_fixed 4113301635.0805855
```

```text
seasonal_naive oracle_top1_rate=0.69485 mse=2.641358e10
dlinear_quito  oracle_top1_rate=0.29000 mse=1.322765e11
patchtst_quito oracle_top1_rate=0.01515 mse=7.465517e11
```

### 6.4 TSMixer lr=3e-4

```text
num_common_windows 20000
best_fixed_expert seasonal_naive
best_fixed_mse 26413584764.11275
oracle_mse 22462335342.67148
oracle_gap_vs_best_fixed 3951249421.441269
```

```text
seasonal_naive oracle_top1_rate=0.69220 mse=2.641358e10
dlinear_quito  oracle_top1_rate=0.26445 mse=1.322765e11
tsmixer_quito  oracle_top1_rate=0.04335 mse=4.432242e11
```

### 6.5 TSMixer lr=3e-4 no RevIN

```text
num_common_windows 20000
best_fixed_expert seasonal_naive
best_fixed_mse 26413584764.11275
oracle_mse 22387161679.456707
oracle_gap_vs_best_fixed 4026423084.656044
```

```text
seasonal_naive oracle_top1_rate=0.69390 mse=2.641358e10
dlinear_quito  oracle_top1_rate=0.28550 mse=1.322765e11
tsmixer_quito  oracle_top1_rate=0.02060 mse=8.445269e11
```

## 7. 结论

PatchTST 的问题不是单纯学习率过高。把 lr 从 `1e-3` 降到 `3e-4` 后，默认容量 PatchTST 仍然严重异常；小模型的 train loss 明显正常，但预测 MSE 仍高达 `1.60e14`。关闭 RevIN 后预测 MSE 降到 `7.47e11`，但 train loss 爆炸，说明关闭 RevIN 不是稳定修复。

TSMixer 在 `lr=3e-4` 下 train loss 很低，但预测 MSE 仍为 `4.43e11`，oracle top1 仅 `4.335%`。关闭 RevIN 后更差。TSMixer 当前配置可作为弱互补候选观察，但不值得直接扩到全量。

DLinear 在 20k/e5 中继续稳定承担互补角色，oracle top1 约 26-30% 的区间内，仍是当前最值得进入 OOF 规划的训练型专家。

当前更像是模型适配/尺度恢复/预测输出口径问题，而不是简单“训练轮数不足”。尤其 PatchTST 需要检查：

- Quito `PatchTST.predict()` 输出是否与项目 wide prediction schema 尺度一致；
- RevIN 的 inverse transform 是否在 predict/loss 路径中按预期工作；
- 当前 sample-channel 单变量输入是否与 Quito PatchTST 原始假设匹配；
- 是否需要按 train-set scaler 或 per-window scaler 显式归一化，而不是只依赖 RevIN；
- 是否需要独立 valid loss，而不是只看 train final loss。

## 8. 下一步

建议进入 Stage 1.4e：专家池重选与 PatchTST 适配审计。

建议范围：

1. 对 PatchTST 做代码级输出尺度审计，构造 toy scale 输入，检查 `loss()` 与 `predict()` 是否完成一致尺度恢复。
2. 重读 TimeFuse、QuitoBench、TimeRecipe、VisMoE 的专家选择逻辑，优先补频域、多尺度、统计强 baseline，而不是继续盲调 PatchTST。
3. 保留 DLinear + seasonal naive 作为当前稳定基础专家。
4. TSMixer 暂不进入 OOF，除非后续模型池比较显示其在特定 cell 有独特贡献。

