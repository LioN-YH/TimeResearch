# 复现协议审计与结果复盘

更新时间：2026-06-08

本文暂停继续堆模型实验，专门解释当前 `matrix50k_v1` 复现结果为什么与 QuitoBench 论文 Table 24 差距大，甚至出现 PatchTST / DLinear / SNaive 排序倒挂。

## 1. 审计对象

当前 clean expert matrix：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/seasonal_naive_period6__matrix50k_v1
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/dlinear__matrix50k_v1_e20_std
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/patchtst__matrix50k_v1_e20_std
```

核心对照来源：

- 论文：`https://arxiv.org/abs/2603.26017`，Table 24。
- 官方数据加载：`quito/quito/datasets.py`
- 官方训练器：`quito/quito/trainers/base.py`, `quito/quito/trainers/trainers.py`
- 官方模型与指标：`quito/quito/models/base.py`, `quito/quito/metrics.py`
- 官方配置：`quito/configs/evaluate/dlinear/96_48_S.yaml`, `quito/configs/evaluate/patchtst/96_48_S.yaml`
- 我们的专家构建：`tools/quitobench_framework_expert_cache.py`, `tools/quitobench_lightweight_expert_cache.py`

## 2. 官方流程确认

### 2.1 数据划分

官方 `TimeSeriesDataset.process_raw_df()` 在有 `global_test_point='2023-07-28 00:00:00'` 时：

- 先定位 `train_valid_size = index(global_test_point)`；
- 再按 `train_ratio=0.7, valid_ratio=0.2` 划分 global test point 之前的数据；
- test 是 global test point 之后的数据；
- valid/test 会向前 overlap 一个 `seq_len`，保证 valid/test 样本有历史窗口。

这与我们 `split_context_policy=quito_overlap` 的设计方向一致。

### 2.2 标准化

官方代码在 `process_raw_df()` 中执行：

```python
mean = np.mean(data[:, :train_size, :], axis=1, keepdims=True)
std = np.std(data[:, :train_size, :], axis=1, keepdims=True) + 1e-8
data = (data - mean) / std
```

结论：

- scaler 是 train segment fit，不是全数据 fit。
- scaler 粒度是 `item_id x channel`，分别在 `hour/min` 文件内部处理。
- 没有看到 per-cell scaler 或 global scaler。
- `normalize=False` 在当前 Quito 源码中不会阻止 `process_raw_df()` 标准化，只会影响 `inverse_transform()`，因此复用官方口径时应显式 `normalize=True`。

我们当前 DLinear/PatchTST cache 的 `--train-set-standardize` 已经使用 Quito `TimeSeriesDataset` 的 train segment scaler，方向正确。

### 2.3 训练 loss

官方配置 `loss: mse`，`BaseModel.setup_loss_fn()` 绑定 `nn.MSELoss`。

官方 `TimeSeriesModel.loss()` 在数据集已经标准化后的 `x/y` 上训练；如果 `revin=True`，模型内部再对每个窗口做 RevIN 归一化，预测后与同一 RevIN 尺度下的 `y` 计算 loss。

我们当前训练：

- DLinear/PatchTST 也是 MSE loss。
- 输入/target 先按 Quito train segment scaler 标准化。
- 模型内部 `revin=True`。

但是我们的训练数据不是官方 dense full training set，而是 `matrix50k_v1` 中的 train rows：`30,134` windows。官方 full Quito dense rolling window 是千万级样本，训练分布和样本量不一致。

### 2.4 测试指标

官方 `evaluate.py` 调用 `model.eval_step()`，`TimeSeriesModel._eval_step()` 直接对当前张量尺度计算 `cal_score(metric, y_pred, y_true)`；没有看到测试前 `inverse_transform()`。因此论文 Table 24 的 MAE 更接近 normalized-scale MAE，而不是 raw-scale MAE。

我们当前同时有两种口径：

- raw-scale audit：预测 inverse transform 后与 raw target 比较。
- normalized-scale audit：按 Quito train segment scaler 回到 normalized scale 后比较。

与论文 Table 24 对齐时，应优先看 normalized-scale MAE。raw-scale MAE 可以用于业务数值误差和 routing 风险，但不能直接拿来对 Table 24。

### 2.5 early stopping / checkpoint

官方 `96_48_S` 配置：

- `num_epochs: 20`
- `eval_epochs: 1`
- `enable_early_stopping: false`
- `checkpointing.enable_checkpoints: true`
- `save_epochs: 1`
- `resume.checkpoint_path` 在 evaluate 配置中列出 `ckpt_0.pkl`, `ckpt_1.pkl`, `ckpt_2.pkl`

官方配置并不是“按 validation MSE 选 best checkpoint”的典型 early stopping 流程。训练器里存在 best checkpoint 逻辑，但 `enable_early_stopping=false`，而 evaluate 配置显式评估多个 checkpoint。我们当前专家 cache 训练后直接用最终模型预测，没有复刻官方多 checkpoint 评估/选择逻辑。

## 3. Raw-scale MSE/MAE vs normalized-scale MSE/MAE

raw-scale 指标：

- 在原始数值尺度计算。
- 大量级 item/channel 会主导整体 MSE/MAE。
- 适合回答“真实数值误差有多大”，不适合与 Quito Table 24 直接对齐。

normalized-scale 指标：

- 用 train segment `mean/std` 归一化后计算。
- 更接近官方 `TimeSeriesDataset + cal_score` 口径。
- 跨 item/channel 更可比，但 small-std / near-constant 序列会放大误差，尤其是 MSE。

后续建议两者都保留，但用途分开：

- 论文复现/协议对齐：normalized MAE/MSE 为主，优先 MAE。
- routing/gate 训练：不能直接用未处理的 normalized MSE，需先确定 outlier policy；MAE/Huber/winsorized MSE 更稳。
- 业务尺度解释：raw MAE/MSE 作为补充。

## 4. 当前 50k 按 TSF regime 对齐 Table 24

已输出对照表：

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/protocol_audit_matrix50k_v1/table24_current_cell_mae_comparison.csv
```

下表使用当前 normalized MAE 与论文 Table 24 MAE 比较。raw MAE 不与 Table 24 直接比较。

| TSF regime | expert | windows | current normalized MAE | Table 24 MAE | 差值 |
| --- | --- | ---: | ---: | ---: | ---: |
| HIGH HIGH HIGH | DLinear | 10152 | 0.160 | 0.189 | -0.029 |
| HIGH HIGH HIGH | PatchTST | 10152 | 0.302 | 0.173 | +0.129 |
| HIGH HIGH HIGH | SNaive | 10152 | 0.282 | 0.898 | -0.616 |
| HIGH HIGH LOW | DLinear | 8055 | 0.442 | 0.418 | +0.024 |
| HIGH HIGH LOW | PatchTST | 8055 | 1.099 | 0.367 | +0.732 |
| HIGH HIGH LOW | SNaive | 8055 | 0.575 | 0.674 | -0.099 |
| HIGH LOW HIGH | DLinear | 4043 | 0.662 | 0.318 | +0.344 |
| HIGH LOW HIGH | PatchTST | 4043 | 1.309 | 0.191 | +1.118 |
| HIGH LOW HIGH | SNaive | 4043 | 0.507 | 0.432 | +0.075 |
| HIGH LOW LOW | DLinear | 3883 | 1.432 | 0.805 | +0.627 |
| HIGH LOW LOW | PatchTST | 3883 | 2.832 | 0.669 | +2.163 |
| HIGH LOW LOW | SNaive | 3883 | 0.664 | 0.417 | +0.247 |
| LOW HIGH HIGH | DLinear | 6959 | 0.501 | 0.260 | +0.241 |
| LOW HIGH HIGH | PatchTST | 6959 | 0.931 | 0.213 | +0.718 |
| LOW HIGH HIGH | SNaive | 6959 | 0.579 | 1.103 | -0.524 |
| LOW HIGH LOW | DLinear | 8917 | 0.578 | 0.318 | +0.260 |
| LOW HIGH LOW | PatchTST | 8917 | 1.031 | 0.256 | +0.775 |
| LOW HIGH LOW | SNaive | 8917 | 0.714 | 0.890 | -0.176 |
| LOW LOW HIGH | DLinear | 4020 | 0.293 | 0.215 | +0.078 |
| LOW LOW HIGH | PatchTST | 4020 | 0.415 | 0.163 | +0.252 |
| LOW LOW HIGH | SNaive | 4020 | 0.222 | 0.358 | -0.136 |
| LOW LOW LOW | DLinear | 3971 | 0.768 | 0.465 | +0.303 |
| LOW LOW LOW | PatchTST | 3971 | 1.376 | 0.392 | +0.984 |
| LOW LOW LOW | SNaive | 3971 | 0.528 | 0.604 | -0.076 |

直接观察：

- PatchTST 在 8/8 cell 都显著差于 Table 24。
- DLinear 在部分 cell 接近 Table 24，但 trend/lowF 相关 cell 明显更差。
- SNaive 在多个 high-seasonality cell 反而远好于 Table 24，这是异常信号，不应解释为 SNaive 真优于论文模型。

## 5. 高概率导致差距的部分

### 5.1 当前 PatchTST 配置不是官方配置

官方 `quito/configs/evaluate/patchtst/96_48_S.yaml`：

```yaml
model:
  model_name: PatchTST
  patch_len: 16
  d_model: 128
  e_layers: 4
```

未覆盖参数走 `PatchTSTModelConfig` 默认值：

- `n_heads=8`
- `d_ff=2048`
- `dropout=0.05`
- `fc_dropout=0.05`
- `revin=True`

当前 cache manifest：

```json
{
  "patch_len": 16,
  "stride": 8,
  "d_model": 128,
  "d_ff": 256,
  "n_heads": 4,
  "e_layers": 2,
  "dropout": 0.05,
  "fc_dropout": 0.05,
  "revin": true
}
```

这是重大配置偏差。PatchTST 当前偏弱，不能视为官方 PatchTST 复现失败。

### 5.2 当前 DLinear learning rate 不同

官方 `dlinear/96_48_S.yaml`：

```yaml
batch_size: 128
learning_rate: 0.0001
scheduler: cosine
scheduler_kwargs:
  T_max: 50
  eta_min: 1e-05
seed: 16
drop_last: true
```

当前 DLinear cache：

```json
{
  "batch_size": 256,
  "learning_rate": 0.001,
  "scheduler": "cosine",
  "eta_min": 1e-05,
  "random_seed": 20260607
}
```

DLinear 虽然在 raw-scale 上是 best fixed，但这仍然不是官方协议。

### 5.3 当前训练集是抽样 30k train windows，不是官方 dense full train

当前 registry：

- total windows = 50,000
- train = 30,134
- valid = 8,192
- test = 11,674

官方 full Quito dense rolling window 是千万级样本。我们当前训练型专家只看了很小的抽样训练集，并且三种 split 都参与最终 audit。论文 Table 24 是 1,290 items × 10 models × 18 configs 的评估实例聚合，不是这个 50k sample-channel window matrix。

因此当前结果最多是 routing cache smoke / matrix 诊断，不是论文复现实验。

### 5.4 当前评估聚合口径不同

论文 Table 24 是按 TSF regime 的 mean MAE，描述中对应 232,200 evaluation instances。当前表是 sampled windows 的 sample-weighted mean MAE，并且 cell 分布不是严格均匀：

```text
highT_highS_highF 10152
highT_highS_lowF   8055
highT_lowS_highF   4043
highT_lowS_lowF    3883
lowT_highS_highF   6959
lowT_highS_lowF    8917
lowT_lowS_highF    4020
lowT_lowS_lowF     3971
```

这会影响总体排名，也会让 dense-window 多的 item/channel 权重更高。

### 5.5 normalized MSE outlier 是真实风险，但不是唯一解释

`docs/MATRIX50K_ERROR_PATHOLOGY_AUDIT.md` 已确认：

- 原始数据 NaN/Inf 为 0。
- DLinear/PatchTST normalized MSE 被 small-std near-constant window 强烈放大。
- DLinear top1 normalized MSE share = 99.07%，PatchTST = 98.61%。

但 Table 24 对齐看的是 MAE 时，PatchTST 仍然显著偏差。因此不能把全部问题归因于 single-window MSE outlier。

## 6. 已确认不是问题的部分

- 三专家 cache 已经共用同一个 `sampled registry`，不是旧的独立 50k cache 交集问题。
- `common_prediction_windows == 50000`，`common_error_windows == 50000`。
- SNaive 当前确实使用 `seasonal_period_override=6`。
- 原始 parquet 数值列没有 NaN/Inf。
- 当前 DLinear/PatchTST `--train-set-standardize` 的 scaler 粒度与 Quito 源码一致：train segment, item/channel。
- 目前没有源码证据支持官方按 TSF cell 单独标准化。

## 7. 需要立刻检查的代码位置

优先级最高：

- `tools/quitobench_framework_expert_cache.py`
  - `_make_patchtst_model()`：当前 CLI 默认 `n_heads=4, d_ff=256, e_layers=2`，与官方配置不一致。
  - `_train_model()`：只训练 registry 中 train split 的抽样 windows，没有 valid evaluation 或 checkpoint selection。
  - `parse_args()`：默认 `learning_rate=0.001`, `batch_size=32`，运行命令覆盖后仍需与官方配置逐项对齐。
- `quito/configs/evaluate/patchtst/96_48_S.yaml`
  - 官方 PatchTST 口径基准。
- `quito/configs/evaluate/dlinear/96_48_S.yaml`
  - 官方 DLinear 口径基准。
- `quito/quito/datasets.py`
  - `process_raw_df()` 的 split / scaler / overlap 是复现协议锚点。
- `quito/quito/models/base.py`
  - `TimeSeriesModel.loss()` 与 `_eval_step()` 决定 loss/metric 的尺度。
- `tools/quitobench_normalized_oracle_audit.py`
  - 用于 Table 24 对齐的 normalized metric 审计。

## 8. 下一步最小排查计划

不要新增模型，也不要进入 visual/gate。先做下面四个最小复现实验/检查。

### P0：协议对齐 smoke，不重训大模型

目标：确认官方 Quito loader/trainer 在本地能跑通，并输出 normalized MAE。

- 用官方 `configs/evaluate/dlinear/96_48_S.yaml` / `patchtst/96_48_S.yaml` 中的模型参数构造最小训练命令。
- 数据仍可先用小 `ids` 或极小 max items，但必须通过 `TimeSeriesDataset` dense loader，而不是 registry 抽样 loader。
- 验证 eval metric 不 inverse transform。

### P1：重建 matrix50k 的 official-config 专家 cache

只在同一个 `matrix50k_v1` 上重建 DLinear/PatchTST，一次性排除配置偏差：

- PatchTST：`e_layers=4, n_heads=8, d_ff=2048, d_model=128, patch_len=16, lr=1e-4, batch=128, seed=16, drop_last=true, scheduler=cosine, eta_min=1e-5`
- DLinear：`lr=1e-4, batch=128, seed=16, drop_last=true, scheduler=cosine, eta_min=1e-5`

注意：这仍然不是论文复现，只是判断“当前 PatchTST 差”是否主要来自配置不一致。

### P2：官方 dense subset 复现实验

如果 P1 后仍然严重倒挂，再跑一个小但更接近官方训练分布的 dense subset：

- 选少量 item ids，但每个 item 使用 full dense rolling train windows。
- 用官方 config 和 `TimeSeriesDataset`。
- 只评估 test split normalized MAE。
- 按 item/cell 聚合，而不是按 sampled window 聚合。

### P3：评估聚合口径复核

在已有 predictions 上补充：

- test-only normalized MAE；
- per-item mean MAE 后再按 cell macro average；
- per-cell mean MAE 与 Table 24 对齐；
- drop-top-k / winsorized MSE 只作为 MSE 稳定性诊断，不替代 MAE 复现。

## 9. 当前结论

当前结果不能作为“PatchTST 在 QuitoBench 上弱于 DLinear/SNaive”的结论。

更准确的判断是：

- 数据标准化方向基本正确，缺失值也不是主因。
- 当前 raw/normalized 双口径审计有价值，但 raw-scale 不能对齐论文 Table 24。
- 当前 PatchTST 与官方配置存在重大差异，是最需要优先修正的复现协议问题。
- 当前训练样本量和训练分布与官方 dense full train 差距很大，是第二个高概率原因。
- 当前评估聚合口径与论文不同，是第三个高概率原因。
- 在这些问题修正前，不建议继续分析 visual/gate 或扩大专家集合。

## 10. 2026-06-08 追加：matrix50k official-config 排查

用户确认后，已先在当前 `matrix50k_v1` 上做了一次最小 official-config 排查，没有重建 balanced registry，也没有启动 dense full 训练。

### 10.1 Runner 最小修正

为避免“重跑仍不对齐官方配置”，已对 `tools/quitobench_framework_expert_cache.py` 做最小参数透传：

- 增加 `decoder_label_len` 到 DLinear / PatchTST / TSMixer config，并传入 Quito model config。
- 增加 CLI `--decoder-label-len`。
- 增加 CLI `--scheduler-t-max`，允许 cosine scheduler 使用官方 `T_max=50`，而不是默认 `T_max=epochs`。
- 增加 CLI `--random-seed`，允许使用官方 `seed=16`。

验证：

```text
conda run -n quito python -m pytest \
  tests/test_quitobench_dlinear_expert_cache.py \
  tests/test_quitobench_expert_cache_audit.py \
  tests/test_quitobench_normalized_oracle_audit.py \
  tests/test_quitobench_oracle_target_audit.py -q

32 passed in 13.65s
```

### 10.2 新 cache

新建 DLinear official-config cache：

```text
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/dlinear__matrix50k_v1_e20_official96_48S_std
```

关键参数：

```json
{
  "seq_len": 96,
  "decoder_label_len": 48,
  "pred_len": 48,
  "epochs": 20,
  "batch_size": 128,
  "learning_rate": 0.0001,
  "scheduler": "cosine",
  "scheduler_t_max": 50,
  "eta_min": 1e-05,
  "random_seed": 16,
  "train_set_standardize": true,
  "drop_last": true
}
```

新建 PatchTST official-config cache：

```text
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/patchtst__matrix50k_v1_e20_official96_48S_std
```

关键参数：

```json
{
  "seq_len": 96,
  "decoder_label_len": 48,
  "pred_len": 48,
  "patch_len": 16,
  "stride": 8,
  "d_model": 128,
  "d_ff": 2048,
  "n_heads": 8,
  "e_layers": 4,
  "epochs": 20,
  "batch_size": 128,
  "learning_rate": 0.0001,
  "scheduler": "cosine",
  "scheduler_t_max": 50,
  "eta_min": 1e-05,
  "random_seed": 16,
  "train_set_standardize": true,
  "drop_last": true
}
```

注意：两者仍只训练当前 registry 中的 `30134` 个 train windows，因此仍不是论文 full dense 复现。

### 10.3 Cache audit

```text
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1_official96_48S
```

结果：

```text
common_prediction_windows = 50000
common_error_windows = 50000
expert_ids = dlinear_quito, patchtst_quito, seasonal_naive
```

### 10.4 Raw-scale audit

```text
outputs/vision_ts_routing/oracle_audit/matrix50k_v1_official96_48S
```

| expert | raw MSE | raw MAE | top1 rate |
| --- | ---: | ---: | ---: |
| DLinear | 2.152583e+11 | 39708.208594 | 0.43654 |
| PatchTST | 6.511047e+11 | 67042.399584 | 0.05880 |
| SNaive | 1.463885e+12 | 69592.302286 | 0.50466 |

raw-scale best fixed 仍是 DLinear，并且相比旧 DLinear cache 有改善。

### 10.5 Normalized-scale audit

```text
outputs/vision_ts_routing/oracle_audit/matrix50k_v1_official96_48S_normalized
outputs/vision_ts_routing/oracle_audit/matrix50k_v1_official96_48S_protocol_delta/normalized_expert_metric_delta.csv
```

| expert | old normalized MSE | new normalized MSE | old normalized MAE | new normalized MAE | top1 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| SNaive | 1.459632 | 1.459632 | 0.510185 | 0.510185 | -0.02572 |
| DLinear | 127.947743 | 55.972841 | 0.525655 | 0.451537 | +0.10362 |
| PatchTST | 736.788875 | 361.459653 | 1.020283 | 0.704657 | -0.07790 |

解释：

- official-config 明显改善 DLinear 和 PatchTST，说明之前的参数错配确实是问题。
- DLinear normalized MAE 已优于 SNaive：`0.451537` vs `0.510185`。
- normalized MSE best fixed 仍是 SNaive，因为 small-std outlier 仍强烈支配 MSE。
- PatchTST 从 `1.020283` 改善到 `0.704657`，但仍明显弱于 DLinear/SNaive 的 MAE；因此“配置错配”只能解释一部分 PatchTST 异常。

### 10.6 更新后的判断

已确认不是问题：

- 三专家同 registry 对齐。
- `96/48/S` 的 history/horizon 方向正确。
- train-set item/channel scaler 方向正确。
- PatchTST/DLinear 的一部分协议参数现在已能显式对齐。
- `tools/quitobench_framework_expert_cache.py` 的 `QUITO_ROOT` 审计路径已修正为本仓库内 `quito/`，避免后续 manifest 继续错误显示 `quito_has_* = false`。

高概率仍导致差距的部分：

1. 当前训练数据仍是 sampled registry 的 `30134` train windows，不是官方 dense rolling full train。
2. 当前评估仍是 sample-window weighted，不是论文的 item/config/regime 聚合。
3. 当前 matrix 的 cell 分布不均匀，适合 smoke，不适合最终 balanced benchmark/gate 训练。
4. PatchTST 对 sampled sparse training 可能更敏感；仅靠官方模型超参没有恢复论文排序。

下一步最小计划应改为：

1. 不继续调 PatchTST 超参。
2. 做一个 official `TimeSeriesDataset` dense loader smoke：少量 item，但每个 item 用 dense rolling train windows。
3. 在 dense smoke 上同时跑 DLinear/PatchTST/SNaive 的 normalized test MAE。
4. 如果 dense smoke 中 PatchTST 恢复，再重建 `matrix50k_cellbalanced_v1` 用于 routing/gate。
5. 如果 dense smoke 仍异常，再审计 wrapper 与 Quito trainer/evaluate 聚合差异，而不是继续增加模型。

### 10.7 official-config 后按 TSF regime 对齐 Table 24

新增输出：

```text
outputs/vision_ts_routing/oracle_audit/matrix50k_v1_official96_48S_protocol_delta/table24_current_cell_mae_comparison_official96_48S.csv
```

下表使用 `official96_48S` 参数版本的 normalized MAE。它比第 4 节旧参数表更适合作为当前 matrix50k 诊断依据。

#### 10.7.1 总体汇总

`official96_48S` 的 50k common windows 上，normalized 口径的整体结果如下：

| 指标 | seasonal_naive | dlinear_quito | patchtst_quito |
| --- | ---: | ---: | ---: |
| MSE | 1.459632 | 55.972841 | 361.459653 |
| MAE | 0.510185 | 0.451537 | 0.704657 |
| oracle top1 rate | 0.50466 | 0.43654 | 0.05880 |

补充解释：

- normalized MAE 下，`dlinear_quito` 已优于 `seasonal_naive`，是当前 official-config 的 best fixed expert。
- normalized MSE 仍由 `seasonal_naive` 领先，说明 small-std windows 对平方误差仍有强影响。
- `patchtst_quito` 在 official-config 下明显改善，但仍没有恢复到 overall MAE 最优。

#### 10.7.2 按 TSF cell 的 normalized MAE

| TSF regime | expert | windows | current normalized MAE | Table 24 MAE | 差值 |
| --- | --- | ---: | ---: | ---: | ---: |
| HIGH HIGH HIGH | DLinear | 10152 | 0.130 | 0.189 | -0.059 |
| HIGH HIGH HIGH | PatchTST | 10152 | 0.227 | 0.173 | +0.054 |
| HIGH HIGH HIGH | SNaive | 10152 | 0.282 | 0.898 | -0.616 |
| HIGH HIGH LOW | DLinear | 8055 | 0.360 | 0.418 | -0.058 |
| HIGH HIGH LOW | PatchTST | 8055 | 0.700 | 0.367 | +0.333 |
| HIGH HIGH LOW | SNaive | 8055 | 0.575 | 0.674 | -0.099 |
| HIGH LOW HIGH | DLinear | 4043 | 0.690 | 0.318 | +0.372 |
| HIGH LOW HIGH | PatchTST | 4043 | 0.840 | 0.191 | +0.649 |
| HIGH LOW HIGH | SNaive | 4043 | 0.507 | 0.432 | +0.075 |
| HIGH LOW LOW | DLinear | 3883 | 1.144 | 0.805 | +0.339 |
| HIGH LOW LOW | PatchTST | 3883 | 1.855 | 0.669 | +1.186 |
| HIGH LOW LOW | SNaive | 3883 | 0.664 | 0.417 | +0.247 |
| LOW HIGH HIGH | DLinear | 6959 | 0.436 | 0.260 | +0.176 |
| LOW HIGH HIGH | PatchTST | 6959 | 0.713 | 0.213 | +0.500 |
| LOW HIGH HIGH | SNaive | 6959 | 0.579 | 1.103 | -0.524 |
| LOW HIGH LOW | DLinear | 8917 | 0.484 | 0.318 | +0.166 |
| LOW HIGH LOW | PatchTST | 8917 | 0.776 | 0.256 | +0.520 |
| LOW HIGH LOW | SNaive | 8917 | 0.714 | 0.890 | -0.176 |
| LOW LOW HIGH | DLinear | 4020 | 0.299 | 0.215 | +0.084 |
| LOW LOW HIGH | PatchTST | 4020 | 0.338 | 0.163 | +0.175 |
| LOW LOW HIGH | SNaive | 4020 | 0.222 | 0.358 | -0.136 |
| LOW LOW LOW | DLinear | 3971 | 0.647 | 0.465 | +0.182 |
| LOW LOW LOW | PatchTST | 3971 | 0.871 | 0.392 | +0.479 |
| LOW LOW LOW | SNaive | 3971 | 0.528 | 0.604 | -0.076 |

#### 10.7.3 按 subset 聚合的表现

| subset | windows | best fixed expert | best fixed MSE | best fixed MAE | oracle top1 MAE |
| --- | ---: | --- | ---: | ---: | ---: |
| hour | 31617 | dlinear_quito | 0.636857 | 0.324711 | 0.291470 |
| min | 18383 | seasonal_naive | 1.415113 | 0.457204 | 0.412695 |

结论：

- official-config 后 PatchTST 已明显改善，但仍在 8/8 cell 差于 Table 24。
- DLinear 在 `HIGH HIGH *` 两个高 trend/high seasonality cell 已接近或优于 Table 24，但在 low seasonality、low forecastability cell 仍偏差较大。
- SNaive 在多个 high-seasonality cell 远好于 Table 24，说明当前 matrix50k 的窗口/聚合/训练分布仍不能解释为论文复现。
- 因此下一步不应继续调模型超参，而应优先验证 dense rolling train/eval 协议与聚合口径。

### 10.8 对两个当前问题的直接回答

**matrix50k_v1 各 cell 分布不均匀是否 OK？**

作为 smoke / pipeline sanity 是 OK 的，因为它覆盖了 8 个 TSF regime、两个 subset、train/valid/test，并且三专家完全同一 registry。但它不适合作为最终论文复现或 gate 主训练集：`highT_highS_highF=10152`，`highT_lowS_lowF=3883`，最大/最小 cell 约 2.6 倍；同时 split-subset-cell 里有些组合极少，例如 test/hour/highT_lowS_lowF 只有 9 个窗口。后续如果用于 routing/gate，建议重建 `matrix50k_cellbalanced_v1` 或至少在 loss/metric 聚合时做 per-cell/per-item reweighting。

**当前 history/horizon 是否就是 96/48？其他参数是否对齐？**

`matrix50k_v1` registry 和当前 official-config cache 的 `seq_len=96, pred_len=48` 是对的；official config 还要求 `decoder_label_len=48`，这一点已经加入 runner 并在 `*_official96_48S_std` cache manifest 中确认。PatchTST official-config 版本还对齐了 `patch_len=16, d_model=128, d_ff=2048, n_heads=8, e_layers=4, lr=1e-4, batch=128, seed=16, scheduler T_max=50, eta_min=1e-5, drop_last=true`。DLinear official-config 版本对齐了 `lr=1e-4, batch=128, seed=16, scheduler T_max=50, eta_min=1e-5, drop_last=true`。

仍未完全对齐官方的部分是训练/评估协议：我们没有使用官方 dense full train loader，没有保存并评估多个 checkpoint，也没有复刻官方 `evaluate.py` 的 per-user/per-item 聚合。因此当前 official-config cache 只能叫“参数对齐的 matrix smoke”，不能叫论文复现。

### 10.9 dense loader smoke

为进一步排查“是否是 sampled registry 本身导致倒挂”，已用官方 `TimeSeriesDataset` 直接做一个小 dense smoke：

- item 选择：`hour=110030`，`min=18393`
- 训练脚本：`tools/quitobench_dense_smoke.py`
- 训练/评估：2 epochs，`seq_len=96`, `pred_len=48`, `decoder_label_len=48`
- 口径：官方 `finetune/evaluate` 配置，dense rolling windows，非 registry 抽样

结果：

| model | test MSE | test MAE |
| --- | ---: | ---: |
| DLinear | 0.255948 | 0.258770 |
| PatchTST | 0.123296 | 0.149131 |
| SNaive period=6 | 0.708998 | 0.372927 |

这组 smoke 不能替代正式复现，但它说明：

- 在官方 dense loader 下，PatchTST 并没有天然倒挂；
- 在同一批 dense windows 上，排序恢复为 PatchTST > DLinear > SNaive；
- 当前 `matrix50k_v1` 的倒挂更像是 sampled sparse registry / 训练分布 / 聚合口径问题，而不是“模型结构本身失效”；
- 后续如果要继续解释论文差距，应优先检查 dense/full protocol，而不是继续堆 PatchTST 超参。

实现备注：官方 SNaive no-training evaluate wrapper 在本地 smoke 中没有及时产出进度，因此 SNaive 这里使用 `tools/quitobench_dense_smoke.py --mode snaive-direct` 直接在 Quito `TimeSeriesDataset` 的 dense test windows 上计算 `seasonal_period=6` 预测误差，仍然是 normalized-scale dense-loader 口径。

### 10.10 8-cell Stage0.7 代表项 dense smoke

为检验“dense rolling loader + 更覆盖 TSF regime 的 item subset”是否足以解释 matrix50k 倒挂，又用 Stage0.7 channel-STL exact-match 规则选了每个官方 TSF cell 一个代表 item：

| official TSF cell | subset | item_id |
| --- | --- | ---: |
| highT_highS_highF | hour | 104560 |
| highT_highS_lowF | hour | 101960 |
| highT_lowS_highF | min | 11217 |
| highT_lowS_lowF | hour | 104313 |
| lowT_highS_highF | min | 17393 |
| lowT_highS_lowF | hour | 108117 |
| lowT_lowS_highF | min | 6461 |
| lowT_lowS_lowF | min | 20064 |

选择依据：

- 使用 `outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv`；
- 要求 `official_tsf_cell == paper_like_tsf_cell`；
- 每个 cell 取距离 `tau=0.4` 边界 margin 最大的 item；
- 该规则用于 smoke 代表项，不等价于最终 dataset 构造规则。

输出路径：

```text
outputs/vision_ts_routing/quito_dense_smoke/cell8_stage07_representatives/
outputs/vision_ts_routing/quito_dense_smoke/cell8_stage07_representatives/dlinear_e5_lr1e3/
outputs/vision_ts_routing/quito_dense_smoke/cell8_stage07_representatives/patchtst_e5_lr1e3/
```

所有模型在同一组 `75,400` 个 dense test windows 上评估，指标仍是 Quito normalized-scale test metric。

| subset | model | train budget | test windows | test MSE | test MAE |
| --- | --- | --- | ---: | ---: | ---: |
| 2-item smoke | PatchTST | e2/lr1e-4 | 18,850 | 0.123296 | 0.149131 |
| 2-item smoke | DLinear | e2/lr1e-4 | 18,850 | 0.255948 | 0.258770 |
| 2-item smoke | SNaive p=6 | direct | 18,850 | 0.708998 | 0.372927 |
| 8-cell reps | SNaive p=6 | direct | 75,400 | 5.446183 | 0.385317 |
| 8-cell reps | DLinear | e2/lr1e-4 | 75,400 | 4.795908 | 0.442782 |
| 8-cell reps | PatchTST | e2/lr1e-4 | 75,400 | 5.831586 | 0.452946 |
| 8-cell reps | DLinear | e5/lr1e-3 | 75,400 | 4.935257 | 0.460124 |
| 8-cell reps | PatchTST | e5/lr1e-3 | 75,400 | 5.229956 | 0.484615 |

解释：

- 2-item dense smoke 能恢复 `PatchTST > DLinear > SNaive`，说明 PatchTST 在本地 Quito dense loader 下并非天然失效。
- 8-cell Stage0.7 代表项混合后没有恢复该排序；SNaive p=6 仍然有最低 MAE。
- DLinear 从 e2/lr1e-4 到 e5/lr1e-3 没有改善，PatchTST 也没有改善，因此当前异常不应继续通过“加 epoch / 调学习率”来解释。
- 8-cell smoke 的结果更像是 item/cell 选择、混合训练、sample-window weighted 聚合或单 item 代表 cell 的偏差，而不是简单的 sparse-vs-dense 二分。

更新后的判断：

- `matrix50k_v1` 仍建议降级为 pipeline smoke；它不适合作为复现结论或 gate 主训练集。
- 但也不能直接用当前 8 个 Stage0.7 代表 item 重建 dense-inspired dataset；单 item/cell 代表性不足，且混合后排序仍异常。
- 下一步应先做 per-cell dense smoke：每次只训练/评估一个 cell 的代表 item，至少输出 PatchTST/DLinear/SNaive 的 normalized MAE；再扩展到每 cell 2-3 个 exact-match item。
- 只有当 per-cell 或每 cell 多 item smoke 能稳定解释 PatchTST/DLinear/SNaive 分布，才进入新的 sampled registry 设计。

当前最小排查计划改为：

1. 对上述 8 个代表 item 做 per-cell dense smoke，优先 e2/lr1e-4，必要时只对异常 cell 复查 e5/lr1e-3。
2. 对每个 cell 追加 2-3 个 Stage0.7 exact-match、高 margin item，检查单 item 代表是否偶然选到 SNaive 友好序列。
3. 汇总 per-cell/per-item MAE，采用 item-macro 和 cell-macro 口径，不再只看 sample-window weighted mean。
4. 若 PatchTST 在多数 cell/item 恢复优势，再设计 `matrix50k_cellbalanced_denseinspired_v1`；若没有恢复，继续审计 Quito 官方 evaluate 聚合和 checkpoint 选择，而不是扩大 gate/visual 实验。

### 10.11 per-cell dense smoke 与 `ids: []` bug

继续拆分 8 个 Stage0.7 代表 item 后，先发现并修复了一个 dense smoke wrapper bug：

- `tools/quitobench_dense_smoke.py::_patch_config()` 过去会把未选择的 subset 写成 `ids: []`；
- Quito `TimeSeriesDataset.process_raw_df()` 中的逻辑是 `if self.ids:`，因此 `ids: []` 不会过滤为空，反而会加载该 subset 的全量 item；
- 这会让“单 hour item”实验意外加载全量 min 数据，导致 SNaive direct 长时间运行且协议不干净；
- 已改为：有 ids 的 subset 保留并设置 ids，没有 ids 的 subset 从 `data.datasets` 删除。

新增测试：

```text
tests/test_quitobench_dense_smoke.py
```

验证：

```text
conda run -n quito python -m pytest tests/test_quitobench_dense_smoke.py -q
2 passed in 4.26s
```

per-cell 汇总输出：

```text
outputs/vision_ts_routing/quito_dense_smoke/per_cell_stage07_representatives/per_cell_metrics.csv
```

所有 cell 均使用单个代表 item、dense test windows、normalized-scale MAE。DLinear/PatchTST 为 e2/lr1e-4，SNaive 为 p=6 direct。

| TSF cell | subset | item_id | SNaive MAE | DLinear MAE | PatchTST MAE | best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| highT_highS_highF | hour | 104560 | 0.077744 | 0.070318 | 0.067199 | PatchTST |
| highT_highS_lowF | hour | 101960 | 0.364597 | 0.291707 | 0.245042 | PatchTST |
| highT_lowS_highF | min | 11217 | 0.206237 | 0.185768 | 0.143245 | PatchTST |
| highT_lowS_lowF | hour | 104313 | 0.037314 | 0.033950 | 0.030800 | PatchTST |
| lowT_highS_highF | min | 17393 | 0.110153 | 0.359768 | 0.111009 | SNaive |
| lowT_highS_lowF | hour | 108117 | 0.751272 | 0.571777 | 0.327224 | PatchTST |
| lowT_lowS_highF | min | 6461 | 1.018801 | 1.456972 | 2.247077 | SNaive |
| lowT_lowS_lowF | min | 20064 | 0.254078 | 0.419530 | 0.330998 | SNaive |

宏平均 MAE：

| model | cell-macro MAE |
| --- | ---: |
| SNaive p=6 | 0.352524 |
| DLinear e2/lr1e-4 | 0.423724 |
| PatchTST e2/lr1e-4 | 0.437824 |

winner count：

| model | best cells |
| --- | ---: |
| PatchTST e2/lr1e-4 | 5 |
| SNaive p=6 | 3 |
| DLinear e2/lr1e-4 | 0 |

解释：

- per-cell 视角下 PatchTST 在 5/8 个代表 cell 最好，说明 8-cell mixed dense smoke 的“PatchTST 整体不佳”不是全局失效。
- SNaive 在 3 个 lowT 相关代表 cell 上最好，尤其 `lowT_lowS_highF/min/6461` 中 PatchTST MAE 很高，强烈拉动混合平均。
- DLinear 在这 8 个单 item 代表上没有 best cell，但多数 cell 处于 SNaive 与 PatchTST 之间。
- 当前结论仍不能支持直接重建大 dataset；下一步应扩展到每 cell 2-3 个 exact-match item，判断这 3 个 SNaive-best cell 是否是单 item 偶然性，还是 Quito cell 内真实结构。

对“通用视觉先验”的影响：

- cell 只用于诊断和采样平衡，不作为模型输入，也不训练 per-cell visual/gate；
- 后续统一 registry 仍应是跨 cell 的一个整体；
- 但在构建统一 registry 前，必须先用 per-cell/per-item 报告确认采样不会被少数 SNaive-friendly item 主导。

### 10.12 下一阶段：专家协议与 Router 数据构造评审

基于当前 dense/sparse 证据，下一阶段不应直接恢复 visual embedding / gate 训练，也不应把 `matrix50k_v1` 当作正式 router 数据集。

新的阶段目标是评审专家协议本身：

- DLinear/PatchTST 是否在统一、平衡、out-of-item 协议下仍有稳定互补性；
- SNaive-best cell 是否来自单 item 偶然性，还是 Quito cell 内稳定结构；
- expert matrix 是否会让 router 学成 item/cell lookup；
- 是否需要替换或扩展专家池，而不是继续维护固定三专家设定。

执行计划已写入：

```text
docs/NEXT_STAGE_EXPERT_PROTOCOL_REVIEW_PLAN.md
```

关键原则：

- per-item / per-cell 只用于诊断，不作为最终专家训练协议；
- 最终专家训练应使用统一 train pool，而不是每 item 单独训练；
- router 训练/验证/测试应尽量 item-disjoint；
- cell label 只用于采样平衡和报告，不进入视觉模型；
- 只有 expert oracle 在 item-disjoint validation/test 上相对 best fixed 有稳定增益，才恢复 visual router 实验。

