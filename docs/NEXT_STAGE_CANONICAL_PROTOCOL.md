# Next Stage Canonical Protocol

> **2026-06-08 更新：** 本文中的 `matrix50k_v1` 和三专家 canonical 设定已降级为 pipeline smoke / 历史候选协议。最近 dense/sparse 审计显示，当前混合训练和聚合口径下 PatchTST/DLinear/SNaive 排序仍不稳定。进入 visual router / gate 前，必须先完成 `docs/NEXT_STAGE_EXPERT_PROTOCOL_REVIEW_PLAN.md` 中的专家协议与数据构造评审。

本文固定下一阶段推荐主线，避免继续沿 Stage 1.4 后的分支盲目扩展。任何新实验如果偏离本文口径，应在实验日志里说明偏离原因。

## 1. 目标

下一阶段只回答一个问题：

> 在固定、可信、可复用的 QuitoBench sample-channel sparse 任务上，视觉 embedding 是否能为异构专家连续自适应融合提供超过 proxy-only baseline 的增益？

暂不追求：

- 官方 Table 3 full dense 复现。
- 新专家大扩展。
- 端到端联合训练。
- official cluster hard router。

## 2. Canonical 数据口径

推荐任务：

```text
dataset: hq-bench/quitobench
revision: 17362dcb
subsets: hour + min
features: S
channels: ind_1 ... ind_5
history_len: 96
pred_len: 48
split_context_policy: quito_overlap
sample_stride: 288
registry: outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506
fixed 50k matrix registry: outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1
```

该 registry 是 all-channel sparse，包含：

```text
total_windows = 220755
train = 133040
valid = 36165
test = 51550
```

已用 `tools/quitobench_registry_audit.py` 生成分布审计：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/audit/
```

关键观察：

- 覆盖 1,290 个 item 和 5 个 channel。
- 8 个 official TSF cell 都覆盖。
- 最小 `split/subset/cell` 组合只有 40 个窗口。
- 最小 item-channel 窗口数为 21。

如果训练 gate 仍过大，可以从该 registry 再派生一个固定 balanced subset，但必须满足：

- train/valid/test 都覆盖。
- hour/min 都覆盖。
- 8 个 official TSF cell 都覆盖。
- item/channel 尽量均衡。
- 派生 sample set 必须有新的 `sample_set_id` 或独立 manifest。
- 分层 quota 不能超过小组合容量；建议先按最小组合容量设计，再用剩余窗口补足。

## 3. Split 规则

固定使用 `quito_overlap`：

```text
train history/target 在 train 段内
valid target 在 valid 段内，history 可向前借 train 尾部
test target 在 test 段内，history 可向前借 valid 尾部
```

这与 Quito `TimeSeriesDataset` 的 valid/test border 逻辑一致，不视为 target 泄漏。

禁止：

- 用 valid/test target 训练专家或 gate。
- 用 target 区间计算 proxy/imageization/visual embedding。
- 把旧 `strict_within_split` registry 与 `quito_overlap` 结果混比。

## 4. 标准化规则

专家训练和 normalized 指标统一使用 Quito train 段 item/channel scaler：

```text
mean/std source: train segment only
granularity: subset + item_id + channel
implementation: quito.datasets.TimeSeriesDataset
```

主流程中启用：

```text
--train-set-standardize
```

报告至少包含两组指标：

| 指标 | 含义 |
| --- | --- |
| normalized-scale MSE/MAE | 标准化尺度，跨 item/channel 更可比 |
| raw-scale MSE/MAE | 反归一化后原始尺度，反映真实数值误差 |

禁止把 raw-scale 和 normalized-scale 指标混在同一排序表中不加说明。

## 5. Expert Pool

第一版只保留三个专家：

| expert | 角色 | 要求 |
| --- | --- | --- |
| SNaive | 强统计 baseline | `seasonal_period=6`，与当前 official alignment sanity 一致 |
| DLinear | 稳定神经/线性专家 | Quito scaler、train split only |
| PatchTST | patch transformer 专家 | Quito scaler、train split only |

暂不纳入：

- TSMixer：早期结果不稳，先不扩展。
- 更多深度模型：当前瓶颈是口径和 gate 评估，不是专家数量。
- Foundation model：成本和接口变量更大，后续再考虑。

## 6. Expert Cache 要求

每个 expert cache 必须包含：

```text
predictions.parquet
errors.parquet
profiling/cell_model_matrix.csv
profiling/oracle_summary.csv
manifest.json
```

manifest 必须明确：

- `sample_set_id`
- `base_registry_id`
- `input_registry_dir`
- `train_set_standardize`
- scaler source
- expert config
- train window count
- split counts
- raw/normalized 输出尺度
- 是否训练神经专家
- 是否运行 visual encoder
- 是否实现 router

预测 cache 必须保证：

```text
(physical_window_id, expert_id) unique
prediction uses history only
target used only for loss/error/oracle
```

已对当前三份 canonical-candidate cache 跑过只读 audit：

```text
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/official_align_h96_p48_allch_stride288_50k/
```

结论：

- 三个单 cache 内部 key 唯一。
- DLinear/PatchTST 的 horizon 与 manifest `pred_len=48` 对齐。
- DLinear/PatchTST 已启用 Quito train segment item/channel scaler。
- SNaive cache 是 raw statistical baseline，manifest 没有 `standardization` 字段。
- 三个 50k cache 的共同窗口只有 23,456 个。

因此，旧三份 `official_align` cache 不能直接视为同一个 50k expert matrix。

2026-06-08 已重建 clean matrix：

```text
registry:
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

expert root:
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

expert set ids:
seasonal_naive_period6__matrix50k_v1
dlinear__matrix50k_v1_e20_std
patchtst__matrix50k_v1_e20_std
```

clean cache audit:

```text
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
```

结果：

- `common_prediction_windows == 50000`
- `common_error_windows == 50000`
- SNaive period=6 抽查通过。
- DLinear/PatchTST 均启用 Quito train-segment standardization。
- 三专家 `yhat_*` 均 finite，无 NaN/inf。

旧 common-window audit 仍作为 reference 保留。

已完成 common-window oracle target audit：

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/common_23456_official_align/
```

关键 raw-scale 结果：

- common windows: 23,456
- best fixed expert: `dlinear_quito`
- best fixed MSE: `1.986668e+11`
- true uniform MSE: `3.179805e+11`
- oracle top1 MSE: `1.799986e+11`
- oracle gap vs best fixed: `1.866819e+10`
- top1 rate: SNaive `0.549667`，DLinear `0.390348`，PatchTST `0.059985`

解释：

- 这批 common windows 有可见 oracle gap，可先支持 gate smoke。
- PatchTST 在 raw-scale common-window 上贡献很弱，不能据此证明它是稳定强专家；normalized-scale 指标仍需补齐。
- 三个 cache 仍不是同一 50k sampled registry，gate smoke 第一版应明确只在 common 23,456 windows 上评估。

clean `matrix50k_v1` 已完成 raw 和 normalized oracle audit：

```text
raw:
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/

normalized:
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1_normalized/
```

raw-scale 结果：

- common windows: `50000`
- best fixed expert: `dlinear_quito`
- best fixed MSE/MAE: `2.626859e+11` / `45414.205961`
- true uniform MSE/MAE: `7.625250e+11` / `58412.098425`
- oracle top1 MSE/MAE: `2.197719e+11` / `36343.860462`

normalized-scale 结果：

- common windows: `50000`
- best fixed expert by MSE: `seasonal_naive`
- SNaive MSE/MAE: `1.459632` / `0.510185`
- DLinear MSE/MAE: `127.947743` / `0.525655`
- PatchTST MSE/MAE: `736.788875` / `1.020283`
- oracle top1 MSE/MAE: `0.786171` / `0.350153`

当前阻塞：

- raw-scale 和 normalized-scale 的 best fixed 排名明显分歧。
- DLinear normalized MAE 与 SNaive 接近，但 normalized MSE 被少数大误差强烈放大。
- 在解释清楚小 std / outlier / subset-cell 分布前，不进入正式 visual/gate 训练。

## 7. Ensemble / Oracle Target

下一阶段必须修正 comparison 口径，至少输出：

| 名称 | 定义 |
| --- | --- |
| best fixed expert | 在评估集合上平均误差最低的单专家 |
| true uniform ensemble | 先平均专家预测，再计算误差 |
| oracle top1 | 每个窗口选择误差最低专家 |
| soft oracle | `softmax(-error / T)` |
| oracle gap | best fixed 与 oracle/learned gate 的差距 |

当前 `uniform_mse_proxy = mean(expert MSE)` 只能保留为诊断字段，不作为 ensemble 主指标。

建议 soft oracle 在 normalized-scale MSE 上生成，避免 raw MSE 尺度过大导致权重塌缩。

## 8. Visual Pipeline

基于同一 canonical registry 重新生成：

```text
view_tensor
visual_embedding_cache
```

第一版 imageization 继续使用 Stage 1.2 三视图：

```text
line_raster
period_fold
fft_power
```

visual embedding cache 输出必须以 `physical_window_id` 为主键，并与 expert errors 一一 join。

第一版不做视觉 encoder 复杂微调：

- 冻结 encoder。
- 只训练 projection/gate。
- 先比较 embedding 是否有用。

## 9. Gate Baseline

第一版 gate 只做小模型：

| baseline | 输入 |
| --- | --- |
| fixed best | 无输入 |
| true uniform | 无输入 |
| proxy-only gate | light proxy + config meta |
| visual-only gate | visual embedding |
| proxy+visual gate | light proxy + visual embedding + config meta |

训练目标：

```text
L_route = KL(soft_oracle || gate_weights)
```

可以另加小权重 prediction loss，但第一版不强制。

评估：

- raw-scale MSE/MAE
- normalized-scale MSE/MAE
- oracle gap reduction
- gate entropy
- expert utilization
- by split/subset/cell 分层结果

## 10. 推荐执行顺序

1. `canonical_registry_audit`
   检查 `96/48/S all-channel stride288` 的 split/subset/cell/item/channel 分布。
   当前第一版审计已生成，后续如果派生 balanced subset，应重新跑同一工具。

2. `canonical_expert_cache_rebuild`
   已完成 clean `matrix50k_v1` 三专家 cache rebuild；后续不要再使用旧三份独立 50k cache 作为同一矩阵。

3. `canonical_oracle_target_audit`
   raw 和 normalized audit 已完成；下一步先排查 normalized MSE outlier/scale pathology，再决定 gate target。

4. `canonical_visual_embedding_cache`
   对同一窗口集合生成三视图 embedding cache。

5. `canonical_gate_smoke`
   训练 proxy-only、visual-only、proxy+visual gate，判断视觉是否有增益。

只有前一步通过 sanity check，才进入下一步。

## 11. 成功/失败判定

继续推进视觉主线的条件：

- proxy+visual 相比 proxy-only 有稳定 oracle gap reduction。
- visual-only 至少在部分 cell/subset 上有可解释增益。
- gate 没有完全塌缩到单一专家。

需要回退重审 imageization/encoder 的条件：

- visual-only 和 proxy+visual 都无收益。
- visual embedding 与 expert preference 几乎无相关。
- embedding cache 受尺度、period 或 split 强烈污染。

需要重审专家池的条件：

- oracle gap 很小，说明专家互补性不足。
- SNaive/DLinear/PatchTST 的 top1 分布极端单一。
- PatchTST 或 DLinear 在 normalized-scale 上仍明显异常。
