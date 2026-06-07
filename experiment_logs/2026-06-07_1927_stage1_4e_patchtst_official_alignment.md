# Stage 1.4e：PatchTST 官方口径对齐与尺度审计

> 后续修正：本日志中的 `train-set standardize` 使用 wrapper-level 全局 scaler，只用于定位尺度/训练口径问题。正式复用 Quito `TimeSeriesDataset` 的 per item/channel train 段 scaler 结果见 `2026-06-07_2000_stage1_4f_quito_dataset_scaler_alignment.md`。

## 1. 目的

Stage 1.4c/1.4d 中 PatchTST 在当前 runner 下出现异常大预测误差，与 QuitoBench 论文中 PatchTST 通常优于 DLinear 的经验不一致。本阶段检查问题是否来自输入尺度、输出尺度、训练预算或官方训练参数未对齐。

本阶段不实现 router/gate，不运行视觉 encoder，不生成 OOF cache，不修改 Quito 上游代码。

## 2. 代码变更

新增 runner 能力：

- `WindowStandardizer`：wrapper-level train split 标准化；
- `--train-set-standardize`：训练/推理内部使用标准化数据，写 prediction cache 前 inverse transform 回原尺度；
- `--drop-last`；
- `--scheduler {none,cosine}`；
- `--eta-min`；
- `--num-workers`；
- `--eval-batch-size`；
- manifest 记录 `standardization`、scheduler、drop_last、final learning rate。

新增诊断工具：

```bash
tools/quitobench_expert_prediction_diagnostics.py
```

该工具读取 expert cache 的 `predictions.parquet`，还原 target，并输出 prediction/target/absolute error 的尺度分布和有限值比例。

相关提交：

```text
d7de3d0 feat: add expert runner train split scaler
c173440 feat: align expert runner training controls
48d19bb feat: add expert prediction scale diagnostics
db2d9e1 fix: allow diagnostics script path execution
```

测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py tests/test_quitobench_expert_prediction_diagnostics.py -q
```

结果：

```text
20 passed
```

runner help smoke 确认包含：

```text
--eval-batch-size
--train-set-standardize
--scheduler {none,cosine}
```

## 3. 512-row smoke

DLinear：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14e_scaler_smoke_512 \
  --stratified-rows 512 \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --device cuda
```

结果：

```text
windows=512
train_windows=168
final_train_loss=0.336928
standardization mean=61999.1328125
standardization std=322260.71875
prediction finite rate=1.0
prediction max=4.5824755e6
```

PatchTST：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model patchtst \
  --expert-set-id patchtst_v1__stage14e_scaler_smoke_512 \
  --stratified-rows 512 \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --device cuda
```

结果：

```text
windows=512
train_windows=168
final_train_loss=0.508517
standardization mean=61999.1328125
standardization std=322260.71875
prediction finite rate=1.0
prediction max=4.2020825e6
```

512-row smoke 中 PatchTST 未再出现 Stage 1.4d 观察到的 `1e9` 级极端预测。

## 4. 20k current-task sanity

任务口径：

```text
seq_len=192
pred_len=96
features=S
stratified_rows=20000
train_windows=7189
epochs=20
batch_size=128
eval_batch_size=128
learning_rate=1e-4
scheduler=cosine
eta_min=1e-5
drop_last=true
train_set_standardize=true
```

DLinear：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler \
  --stratified-rows 20000 \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --num-workers 6 \
  --device cuda
```

结果：

```text
elapsed_seconds=53.00
train_elapsed_seconds=17.62
final_train_loss=0.088199
final_learning_rate=1e-5
standardization mean=79846.8984375
standardization std=955732.0625
prediction finite rate=1.0
prediction max=9.2017344e7
absolute_error mean=29285.6876
```

PatchTST：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model patchtst \
  --expert-set-id patchtst_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler \
  --stratified-rows 20000 \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.0001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --num-workers 6 \
  --device cuda
```

结果：

```text
elapsed_seconds=72.48
train_elapsed_seconds=37.33
final_train_loss=0.071054
final_learning_rate=1e-5
standardization mean=79846.8984375
standardization std=955732.0625
prediction finite rate=1.0
prediction max=1.09980792e8
absolute_error mean=20702.8151
```

## 5. Comparison

对比缓存：

```text
lightweight_v1__seasonal_naive_full
dlinear_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler
patchtst_v1__stage14e_h192_p96_20k_e20_lr1e4_scaler
```

comparison：

```text
comparison_id=stage14e_h192_p96_20k_e20_lr1e4_scaler
common_windows=20000
experts=seasonal_naive,dlinear_quito,patchtst_quito
```

整体结果：

| expert_id | MSE | MAE | oracle top1 | windows |
| --- | ---: | ---: | ---: | ---: |
| `seasonal_naive` | `2.641358e10` | `13306.5680` | `0.8069` | 20000 |
| `patchtst_quito` | `6.776637e10` | `20702.8151` | `0.1587` | 20000 |
| `dlinear_quito` | `2.141623e11` | `29285.6876` | `0.0344` | 20000 |

ensemble summary：

```text
oracle_mse=1.925839e10
best_fixed_expert=seasonal_naive
best_fixed_mse=2.641358e10
oracle_gap_vs_best_fixed=7.155193e9
```

split 层：

| split | best fixed | best fixed MSE |
| --- | --- | ---: |
| train | `patchtst_quito` | `3.496644e10` |
| valid | `seasonal_naive` | `2.213997e10` |
| test | `seasonal_naive` | `1.996298e10` |

cell 层仅 `highT_lowS_lowF` 的 best fixed 为 `patchtst_quito`，其余 7 个 cell 仍为 `seasonal_naive`。

## 6. 结论

Stage 1.4c/1.4d 中 PatchTST 异常差的主要原因不是“PatchTST 模型本身不适合”，而是当前 runner 与官方训练口径不对齐。引入 train-set 标准化、`lr=1e-4`、`epochs=20`、cosine scheduler 和 `drop_last=true` 后，PatchTST：

- 不再出现 `1e9` 级极端预测；
- final train loss 正常；
- 在当前 `192/96/S` 20k sanity 中 MSE 明显优于 DLinear；
- oracle top1 从 Stage 1.4d 的约 `1%` 级别恢复到 `15.87%`。

但 PatchTST 仍未超过 `seasonal_naive`，且 valid/test split 的 best fixed 仍是 `seasonal_naive`。因此现阶段不能直接把 PatchTST 作为强固定专家；它更适合作为互补专家候选继续观察。

## 7. 下一步

不强制立即生成 `96/48/S` registry。原因是当前官方对齐训练已经在 `192/96/S` 上恢复了 QuitoBench 论文方向上的关键相对关系：PatchTST 优于 DLinear。

建议下一步：

1. 保留 DLinear + PatchTST + seasonal naive 作为 Stage 1.4 后续 OOF 设计的候选池，但 OOF 前需要明确训练标准化口径；
2. 若要进一步对齐论文，可单独规划 `96/48/S` registry sanity；
3. 继续重读 TimeFuse/QuitoBench/TimeRecipe/VisMoE，补充覆盖频域、多尺度、统计强 baseline 的专家，而不是只在 PatchTST 上继续调参。
