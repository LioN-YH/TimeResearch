# Stage 1.4g-c：Table 3 口径对齐 sanity

## 1. 目的

用户指出当前结果和论文 Table 3 不一致。本阶段最小化排查已发现的口径差异。

更新说明：本日志最初误以为官方 `features=S` 只预测 `ind_1`。后续复查 `TimeSeriesDataset` 发现该理解不完整：`target: ind_1` 主要把 `ind_1` 移到 numeric columns 第一位；当 `features=S` 时，当前实现会把 `(N, L, C)` reshape 成 `(N*C, L, 1)`，即 QuitoBench 的 5 个通道都会作为单变量样本参与。因此本日志保留 `ind_1-only` 结果作为错误假设下的对照，并新增 all-channel 结果作为更接近官方 S 口径的结论。

已确认的关键差异：

1. 官方 `snaive` 配置使用 `seasonal_period=6`，而不是本项目先前使用的 hour=24/min=144；
2. 官方 `features=S` 在当前实现中是把所有数值通道拆成单变量样本，而不是只预测 `ind_1`；
3. 官方使用 Quito normalize/scaler 和官方 trainer/evaluate 配置。

本阶段仍不是 full exhaustive 官方复现；它使用 `stride=288` sparse registry，但修正 seasonal period、Quito normalize 和主要训练超参。

## 2. 关键官方配置依据

官方配置：

```text
quito/configs/evaluate/snaive/96_48_S.yaml
model:
  model_name: NaiveForecaster
  method: seasonal
  seasonal_period: 6
```

官方 S 任务配置：

```text
target: ind_1
features: S
normalize: true
```

但源码中 `target` 只用于移动列顺序：

```text
if self.target:
    columns = [self.target] + [c for c in df.columns if c != self.target]
```

并且 `features=S` 会拆通道：

```text
if self.features == Features.S:
    data = rearrange(data, 'n l c -> (n c) l 1')
```

PatchTST/DLinear finetune 近似口径：

```text
epochs=5
learning_rate=0.001
batch_size=128
scheduler=cosine
eta_min=0.0001
normalize=true
```

PatchTST 使用：

```text
patch_len=16
d_model=128
e_layers=4
```

## 3. 代码改动

为轻量 baseline 增加：

```text
--seasonal-period-override
```

默认值为 `None`，不影响旧实验；本阶段设置为 `6` 来复现官方 `snaive`。

## 4. Registry：ind_1-only 对照

生成 `96_48_S + ind_1 + stride288` registry：

```bash
/home/user10/miniconda3/envs/quito/bin/python tools/quitobench_window_registry.py \
  --history-len 96 \
  --pred-len 48 \
  --sample-stride 288 \
  --channels ind_1
```

输出：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d/
windows=44151
split_window_counts={'train': 26608, 'valid': 7233, 'test': 10310}
subset_window_counts={'hour': 27918, 'min': 16233}
```

## 5. 执行口径

SNaive：

```bash
tools/quitobench_lightweight_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d \
  --expert-set-id seasonal_naive_period6__official_align_h96_p48_ind1_stride288 \
  --expert-ids seasonal_naive \
  --seasonal-period-override 6
```

DLinear：

```bash
tools/quitobench_framework_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d \
  --expert-model dlinear \
  --expert-set-id dlinear__official_align_h96_p48_ind1_stride288_e5_std \
  --epochs 5 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.0001
```

PatchTST：

```bash
tools/quitobench_framework_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d \
  --expert-model patchtst \
  --expert-set-id patchtst__official_align_h96_p48_ind1_stride288_e5_std \
  --epochs 5 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.001 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.0001 \
  --e-layers 4 \
  --d-model 128 \
  --patch-len 16
```

## 6. ind_1-only 结果

汇总 CSV：

```text
outputs/vision_ts_routing/stage14g_b_official_align_h96_p48_ind1_stride288_summary.csv
```

| model | windows | train_windows | MSE | MAE | MSE/SNaive | MAE/SNaive |
|---|---:|---:|---:|---:|---:|---:|
| SNaive period=6 | 44151 | - | 88.531486 | 4.515134 | 1.000000 | 1.000000 |
| DLinear std | 44151 | 26608 | 31.763939 | 2.816290 | 0.358787 | 0.623744 |
| PatchTST std | 44151 | 26608 | 29.894692 | 2.717031 | 0.337673 | 0.601761 |

## 7. all-channel 官方近似结果

复查代码后，补跑现有 all-channel `96_48_S + stride288` registry：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/
```

口径：

```text
stratified_rows=50000
channels=ind_1..ind_5
SNaive seasonal_period=6
DLinear/PatchTST: Quito normalize, epochs=5, lr=0.001, batch_size=128, cosine eta_min=0.0001
PatchTST: e_layers=4, d_model=128, patch_len=16
```

汇总 CSV：

```text
outputs/vision_ts_routing/stage14g_c_official_align_h96_p48_allch_stride288_50k_summary.csv
```

| model | windows | train_windows | MSE | MAE | MSE/SNaive | MAE/SNaive |
|---|---:|---:|---:|---:|---:|---:|
| SNaive period=6 allch | 50000 | - | 1.130613e+12 | 52585.624575 | 1.000000 | 1.000000 |
| DLinear std allch | 50000 | 20750 | 2.443289e+11 | 39119.452671 | 0.216103 | 0.743919 |
| PatchTST std allch | 50000 | 20750 | 5.367908e+11 | 58989.524368 | 0.474779 | 1.121780 |

## 8. 修正后的结论

all-channel sparse 口径下，DLinear 明显优于官方 period=6 SNaive；PatchTST 的 MSE 优于 SNaive，但 MAE 略差。

因此上一轮“PatchTST/DLinear 弱于 seasonal_naive”的主要原因不是模型天然不好用，而是实验口径错误：

1. 我们把 SNaive 从官方 `seasonal_period=6` 换成了 hour=24/min=144，baseline 被显著加强；
2. 之前 sparse wrapper 没完全使用官方 normalize 和超参；
3. 之前仍不是官方 full-window checkpoint evaluate。

当前仍需注意：本阶段只对齐了 `96_48_S`、all-channel `features=S`、`stride=288` sparse registry。要复现 Table 3，还需要跑官方 full-window 的 finetune/evaluate 或至少复用官方 checkpoint 和 exhaustive test split。
