# Stage 1.4f：复用 Quito TimeSeriesDataset 的标准化口径

## 1. 目的

Stage 1.4e 使用 wrapper-level `WindowStandardizer` 快速验证了 PatchTST 异常与尺度/训练口径有关，但该实现不是 Quito 官方 `TimeSeriesDataset` 的完整标准化流程。本阶段按项目原则修正：尽量复用已有 Quito 代码和框架。

本阶段不实现 router/gate，不运行视觉 encoder，不生成 OOF cache，不修改 Quito 上游代码。

## 2. 代码变更

新增：

- `QuitoWindowScaler`：记录单个 `(subset,item_id,channel)` 的 train 段 mean/std；
- `extract_quito_standardized_series_maps()`：实例化 Quito `TimeSeriesDataset`，复用其 train 段 mean/std，再按本项目 registry 抽取 sample-channel window；
- `inverse_transform_prediction_map()`：按每个 `physical_window_id` 对应的 Quito scaler 将 prediction 逆变换回 raw scale；
- `prepare_model_series_maps()`：统一 runner 的 raw/standardized 数据准备路径。

`--train-set-standardize` 当前正式语义：

```text
原始 parquet
-> Quito TimeSeriesDataset 计算完整 item/channel train 段 mean/std
-> 按 registry row 抽取标准化 history/target
-> 模型训练/推理，默认仍使用 RevIN
-> prediction 按对应 item/channel scaler inverse 回 raw scale
-> 与 raw target 计算误差
```

manifest 中标准化字段示例：

```json
{
  "enabled": true,
  "scope": "quito_timeseries_dataset_train_segment",
  "scaler_granularity": "subset_item_channel",
  "source_dataset": "quito.datasets.TimeSeriesDataset",
  "global_test_point": "2023-07-28 00:00:00",
  "num_window_scalers": 20000
}
```

相关提交：

```text
b377dec feat: reuse quito dataset train scaling for experts
```

## 3. 测试

新增测试覆盖：

- Quito train 段 mean/std 来自完整序列，而不是抽样窗口；
- mean/std 粒度是 `(subset,item_id,channel)`；
- prediction inverse transform 使用 window-specific scaler；
- `prepare_model_series_maps(..., train_set_standardize=True)` 走 Quito adapter。

测试命令：

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py -q
```

结果：

```text
21 passed
```

## 4. 512-row smoke

DLinear：

```text
expert_set_id=dlinear_v1__stage14f_quito_scaler_smoke_512
windows=512
train_windows=168
final_train_loss=1.059160
standardization.scope=quito_timeseries_dataset_train_segment
standardization.scaler_granularity=subset_item_channel
prediction finite rate=1.0
prediction max=4.690305e6
absolute_error mean=25128.9753
```

PatchTST：

```text
expert_set_id=patchtst_v1__stage14f_quito_scaler_smoke_512
windows=512
train_windows=168
final_train_loss=1.099315
standardization.scope=quito_timeseries_dataset_train_segment
standardization.scaler_granularity=subset_item_channel
prediction finite rate=1.0
prediction max=4.14723875e6
absolute_error mean=22488.5937
```

## 5. 20k current-task sanity

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
standardization=Quito TimeSeriesDataset train segment per item/channel
```

DLinear：

```text
expert_set_id=dlinear_v1__stage14f_h192_p96_20k_e20_lr1e4_quito_scaler
elapsed_seconds=93.39
train_elapsed_seconds=17.29
final_train_loss=1.456870
prediction finite rate=1.0
```

PatchTST：

```text
expert_set_id=patchtst_v1__stage14f_h192_p96_20k_e20_lr1e4_quito_scaler
elapsed_seconds=115.11
train_elapsed_seconds=37.63
final_train_loss=2.733177
prediction finite rate=1.0
prediction max=1.1521536e8
absolute_error mean=32414.9108
```

## 6. Comparison

对比缓存：

```text
lightweight_v1__seasonal_naive_full
dlinear_v1__stage14f_h192_p96_20k_e20_lr1e4_quito_scaler
patchtst_v1__stage14f_h192_p96_20k_e20_lr1e4_quito_scaler
```

comparison：

```text
comparison_id=stage14f_h192_p96_20k_e20_lr1e4_quito_scaler
common_windows=20000
```

整体结果：

| expert_id | MSE | MAE | oracle top1 | windows |
| --- | ---: | ---: | ---: | ---: |
| `seasonal_naive` | `2.641358e10` | `13306.5680` | `0.69680` | 20000 |
| `patchtst_quito` | `2.244677e11` | `32414.9108` | `0.09215` | 20000 |
| `dlinear_quito` | `2.741591e11` | `32773.1567` | `0.21105` | 20000 |

ensemble summary：

```text
oracle_mse=2.334481e10
best_fixed_expert=seasonal_naive
best_fixed_mse=2.641358e10
oracle_gap_vs_best_fixed=3.068771e9
```

## 7. 结论

复刻 Quito `TimeSeriesDataset` 标准化后，PatchTST 仍未发散，说明 Stage 1.4c/1.4d 的极端异常确实与训练/尺度口径有关。

但相比 Stage 1.4e 的 wrapper-level 全局 scaler，官方 per item/channel scaler 下 PatchTST 和 DLinear 的整体 MSE 都变差，且仍明显弱于 `seasonal_naive`。PatchTST 的 MSE 略优于 DLinear，但 oracle top1 低于 DLinear，说明二者互补区域不同：

- PatchTST：整体 MSE 略好；
- DLinear：在更多窗口上成为局部 top1；
- seasonal naive：仍是最强 fixed baseline。

因此后续不能基于 Stage 1.4e wrapper-scaler 结果乐观推进 PatchTST。正式路线应以 Stage 1.4f 的 Quito dataset scaler 结果为准。

## 8. 下一步

建议：

1. 将 `--train-set-standardize` 的正式语义固定为 Quito `TimeSeriesDataset` train 段 per item/channel scaler；
2. 在 OOF 前保留 `seasonal_naive + DLinear + PatchTST`，但需要谨慎评估 PatchTST/DLinear 是否值得全量训练；
3. 单独规划 `96/48/S` registry sanity，用官方任务网格判断 PatchTST 是否能在 Quito 官方 horizon 上恢复更强表现；
4. 继续补充频域、多尺度和统计强 baseline，避免专家池只围绕 PatchTST/DLinear。
