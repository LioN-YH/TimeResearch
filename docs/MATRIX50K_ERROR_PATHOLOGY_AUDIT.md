# Matrix50k Error Pathology Audit

日期：2026-06-08

## 1. 背景

clean `matrix50k_v1` 三专家 cache 已经通过 common-window 审计：

```text
registry:
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

expert root:
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

expert cache audit:
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
```

但 raw-scale 和 normalized-scale oracle audit 出现明显分歧：

| metric scale | best fixed | 备注 |
| --- | --- | --- |
| raw-scale MSE | `dlinear_quito` | DLinear 明显优于 SNaive/PatchTST |
| normalized-scale MSE | `seasonal_naive` | DLinear/PatchTST MSE 被极端 outlier 主导 |

本审计目标是确认：

1. 原始数据是否存在 NaN/inf；
2. normalized MSE 是否被少数窗口支配；
3. 异常是否集中在 subset/cell/small std；
4. Quito 官方标准化是否可能是 cell 内标准化；
5. 后续是否应尝试分 cell 标准化。

## 2. 标准化依据

Quito 源码 `quito/quito/datasets.py` 的 `TimeSeriesDataset.process_raw_df()` 使用以下逻辑：

```python
mean = np.mean(data[:, :train_size, :], axis=1, keepdims=True)
std = np.std(data[:, :train_size, :], axis=1, keepdims=True) + 1e-8
data = (data - mean) / std
```

即：

- mean/std 只来自 train segment；
- 粒度是 subset 内每个 item、每个 channel；
- `features=S` 后 item-channel 被 reshape 为独立序列；
- 源码中没有按 TSF cell 分组计算 scaler 的逻辑。

论文侧也强调 QuitoBench 是按 8 个 TSF regime cell 做 balanced benchmark construction 和 per-regime diagnostic reporting，而不是说训练/标准化在 cell 内完成。论文写到使用 TSF cell 做 stratified sampling，并支持 micro/macro aggregation；deep learning models follow tuning/training/evaluation protocol。当前没有看到“按 cell 标准化”的论文或代码证据。

因此，当前 `--train-set-standardize` 复用 Quito train-segment item/channel scaler 是更接近官方代码的口径。分 cell 标准化可以作为诊断对照，但不能声称是官方口径，除非找到新的源码/论文证据。

参考：

- arXiv: https://arxiv.org/abs/2603.26017
- HuggingFace dataset page: https://huggingface.co/datasets/hq-bench/quitobench
- local source: `quito/quito/datasets.py`

## 3. 原始数据缺失值审计

输出：

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/data_missing_summary.json
```

结果：

| subset | rows | items | numeric cols | NaN cells | inf cells | NaN ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| hour | 7,939,052 | 517 | ind_1..ind_5 | 0 | 0 | 0.0 |
| min | 4,563,792 | 773 | ind_1..ind_5 | 0 | 0 | 0.0 |

结论：当前异常不是由原始 parquet 中的 NaN/inf 直接导致。

## 4. Pathology Audit 输出

输出目录：

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/pathology_matrix50k_v1/
```

主要文件：

| 文件 | 用途 |
| --- | --- |
| `per_window_expert_pathology.parquet` | 每个 `(physical_window_id, expert_id)` 的 raw/normalized error、scaler mean/std、metadata |
| `top100_normalized_mse_by_expert.csv` | 每个 expert normalized MSE top100 窗口 |
| `summary_by_expert.csv` | expert 级汇总 |
| `summary_by_expert_subset.csv` | expert x subset 汇总 |
| `summary_by_expert_cell.csv` | expert x official TSF cell 汇总 |
| `summary_by_expert_std_bin.csv` | expert x scaler std bin 汇总 |
| `outlier_contribution.csv` | top-k 窗口贡献率 |

## 5. 核心发现

### 5.1 DLinear/PatchTST normalized MSE 几乎由单个窗口主导

| expert | top1 normalized MSE share | top1 normalized MSE | top1 window |
| --- | ---: | ---: | --- |
| DLinear | 99.07% | 6.337921e+06 | `9fff220180b76e27` |
| PatchTST | 98.61% | 3.632639e+07 | `9fff220180b76e27` |
| SNaive | 9.47% | 6.913076e+03 | different top window |

去掉 top1 后：

| expert | original normalized MSE | drop top1 normalized MSE | original normalized MAE | drop top1 normalized MAE |
| --- | ---: | ---: | ---: | ---: |
| DLinear | 127.947743 | 1.189357 | 0.525655 | 0.484302 |
| PatchTST | 736.788875 | 10.261254 | 1.020283 | 0.924727 |
| SNaive | 1.459632 | 1.321397 | 0.510185 | 0.508605 |

这说明 DLinear 的 normalized MSE 倒挂基本不是总体性能，而是 single-window outlier。PatchTST 也受同一 outlier 强烈影响，但去掉 top1 后仍明显弱于 DLinear/SNaive。

### 5.2 Top outlier 是 near-constant target + tiny train std

Top window:

```text
physical_window_id = 9fff220180b76e27
subset = min
split = test
official_tsf_cell = highT_lowS_lowF
item_id = 4308
channel = ind_5
scaler_mean = 384.000964
scaler_std = 0.031039
target = 384.0 repeated 48 times
SNaive prediction = 384.0 repeated 48 times
```

对应误差：

| expert | raw MSE | raw MAE | normalized MSE | normalized MAE | max abs normalized error |
| --- | ---: | ---: | ---: | ---: | ---: |
| SNaive | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| DLinear | 6105.894405 | 64.191490 | 6.337921e+06 | 2068.122261 | 5017.572053 |
| PatchTST | 34996.511651 | 148.324360 | 3.632639e+07 | 4778.716135 | 11862.615447 |

解释：

- target 是完全常数，SNaive 因为 period=6 直接复制最近值，正好命中。
- scaler std 只有 `0.031039`，所以几十到一百多的 raw deviation 被放大成几千级 normalized error。
- 这类窗口会让 normalized MSE 极端敏感。

### 5.3 异常主要集中在 min / low forecastability / small std

按 subset：

| expert | subset | normalized MSE mean | normalized MAE mean | normalized MSE max |
| --- | --- | ---: | ---: | ---: |
| DLinear | hour | 0.831276 | 0.391944 | 1.432641e+03 |
| DLinear | min | 346.575898 | 0.755624 | 6.337921e+06 |
| PatchTST | hour | 9.076592 | 0.785041 | 6.914374e+04 |
| PatchTST | min | 1988.384329 | 1.424876 | 3.632639e+07 |
| SNaive | hour | 1.485517 | 0.540990 | 1.619084e+03 |
| SNaive | min | 1.415113 | 0.457204 | 6.913076e+03 |

按 cell，DLinear/PatchTST 最大异常都集中在：

```text
highT_lowS_lowF
```

该 cell 对 DLinear/PatchTST 的 normalized MSE 影响远超其他 cell。

### 5.4 small std bin 是主要风险区

全局最低 std bin `(-0.00099999, 19.87]` 覆盖 25,005 windows。该 bin 中：

| expert | normalized MSE mean | normalized MSE max | normalized MAE mean |
| --- | ---: | ---: | ---: |
| DLinear | 254.514179 | 6.337921e+06 | 0.411024 |
| PatchTST | 1459.416994 | 3.632639e+07 | 0.793337 |
| SNaive | 1.202297 | 6.913076e+03 | 0.313489 |

这进一步说明：normalized MSE 在 small-std / near-constant series 上会严重惩罚不精确复制水平值的神经模型。

## 6. 关于“分 cell 标准化”的判断

当前证据不支持“Quito 官方 train-set 标准化是在 cell 内完成的”：

- 源码 scaler 粒度是 item/channel，不读取 cluster/cell。
- 论文描述 cell 的作用是 balanced benchmark construction、macro/micro aggregation、per-regime diagnostic reporting。
- 当前公开/本地代码没有 cell-level scaler。

是否可以尝试分 cell 标准化？

可以，但应定位为诊断实验，而不是官方复现：

1. `cell_global_train_scaler`：每个 `official_tsf_cell` 内，用 train windows 的 raw values 估计一个全局 mean/std。
2. `subset_cell_global_train_scaler`：每个 `(subset, official_tsf_cell)` 内估计 mean/std。
3. 对已有 raw predictions/targets 重新评分，不重训模型，先看 ranking 是否稳定。
4. 如果只是为了 gate target，更建议优先比较 MAE、Huber、winsorized MSE，而不是改变专家训练标准化口径。

注意：分 cell scaler 会引入 TSF label 到误差定义中。路线 1 的 gate 不应把 cell 当硬监督标签；如果使用 cell-level scaler，需要在实验中明确这是 metric diagnostic，不是主训练口径。

## 7. 当前建议

不要立刻进入 visual/gate，也不要马上重训 PatchTST。

优先做三件小事：

1. 在 clean `matrix50k_v1` 上补充 metric policy audit：
   - raw MSE/MAE；
   - normalized MSE/MAE；
   - drop-top-k normalized MSE；
   - winsorized normalized MSE；
   - Huber / MAE-based soft oracle。
2. 对 `highT_lowS_lowF` 和 small-std windows 单独报告 expert ranking。
3. 对 PatchTST 做小规模调参/稳定性检查前，先确认 metric target 不被 single-window outlier 主导。

当前可复用知识：

- 原始数据无 NaN/inf。
- official/Quito scaler 不是 cell-level scaler。
- DLinear normalized MSE 异常主要由单个 near-constant small-std window 主导；去掉 top1 后 DLinear normalized MSE 接近 SNaive。
- PatchTST 即使去掉 top1 后仍偏弱，但问题从“完全倒挂”变成“small-std/lowF 区域不稳 + 模型配置/训练预算可能不足”。
