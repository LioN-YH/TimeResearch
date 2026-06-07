# Stage 1.4：专家预测缓存与框架复用设计

## 1. 设计目的

Stage 1.4 的目标是尽早启动异构专家预测缓存建设，为后续 continuous gate / router 提供稳定的专家预测矩阵和 oracle error 目标。

本阶段要回答的问题：

> 在不实现 router、不训练 gate 的前提下，如何尽量复用 Quito、Time-Series-Library/tslib 等现有时序预测框架，生成与 Stage 1.0 `physical_window_id` 对齐的专家预测缓存，并通过 cell-level profiling 判断专家池是否具备互补性。

本阶段明确不做：

- 不实现 router / gate。
- 不训练视觉 encoder。
- 不把专家和视觉模块做端到端联合训练。
- 不修改 Quito 官方代码作为主路径。
- 不把候选专家池直接等同于最终专家池。
- 不用 TSF cell 标签训练路线 1 的 router；TSF cell 只用于 profiling、解释和路线 2 监督目标。

## 2. 当前上下文

已完成共享资产：

- Stage 1.0 working registry：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
```

- Stage 1.1 / 1.1b light proxy：

```text
outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/sample_channel_proxy.parquet
```

- Stage 1.2 view tensor imageization：

```text
outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1/
```

关键约束：

- `physical_window_id` 是专家预测缓存、proxy、伪图像、oracle error 的统一主键。
- `sample_set_id` 必须保留，便于后续区分 stride96、stratified、smoke 等采样集合。
- 当前第一版 registry 为 `quito_overlap + sample_stride=96`，共 627,430 条 sample-channel 窗口。
- 当前输入粒度继续保持 channel-independent sample-channel window。
- 专家预测只能使用 history 作为输入，target 只能用于评估误差和 oracle label。

## 3. 框架复用原则

Stage 1.4 不应手写完整训练框架。自有代码只负责协议层、缓存层和跨框架适配层。

推荐分工：

| 模块 | 优先策略 |
| --- | --- |
| 数据切窗口径 | 复用 Quito `TimeSeriesDataset` 口径，保持与 Stage 1.0 registry 一致 |
| 专家模型实现 | 优先复用 Quito、Time-Series-Library/tslib 或成熟公开实现 |
| 训练/推理循环 | 优先复用框架 runner；必要时写 thin wrapper |
| 缓存与主键 | 本项目自定义，强制对齐 `physical_window_id` |
| profiling 指标 | 本项目自定义，按 sample、cell、expert 汇总 |
| router / gate | Stage 1.4 不实现 |

具体原则：

1. 不直接改 Quito 官方代码；如果必须改，先写 adapter 或 wrapper。
2. 不把某个框架的内部样本编号作为缓存主键；统一映射回 `physical_window_id`。
3. 不把上游框架输出格式泄漏到后续 gate；统一转换成项目内 prediction cache schema。
4. 每个专家必须记录来源框架、commit/version、配置、训练数据口径、推理 device 和 latency。
5. 候选专家先 profiling，再筛选最终 3-4 个互补性强、成本可控的专家。

## 4. Stage 1.4 分层

### 4.1 Stage 1.4a：极轻量专家缓存

目的：

> 在不启动长训练的情况下，快速建立专家预测缓存 schema、oracle error 计算和 cell-level profiling 脚本。

候选专家：

- `last_value`：最后值延拓。
- `seasonal_naive`：按 period 取历史对应相位。
- `recent_mean`：最近窗口均值。
- `linear_trend`：history 上拟合轻量线性趋势并外推。

这些专家不是最终方法贡献，而是用于：

- 验证 cache schema；
- 验证 `physical_window_id` 对齐；
- 验证 error/oracle 计算；
- 给 neural experts 提供下界参照；
- 在专家训练失败时保留稳定 fallback。

### 4.2 Stage 1.4b：复用框架候选专家

目的：

> 接入 4-8 个代表性 neural / frequency / decomposition 专家，生成统一预测缓存，并评估专家互补性。

候选池建议：

| 专家族 | 候选 | 选择理由 |
| --- | --- | --- |
| decomposition / linear | DLinear, NLinear | 趋势/季节分解强，训练和推理成本低 |
| frequency | FITS 或同类频域模型 | 对强周期、频谱稀疏结构可能有优势 |
| patch transformer | PatchTST | channel-independent patch 建模，是强基线 |
| inverted transformer | iTransformer | token 组织方式不同，适合形成结构差异 |
| MLP / multiscale | TSMixer, TimeMixer | 与 transformer 专家形成 inductive bias 互补 |
| cross-dimension / shape | Crossformer | 作为多维结构专家候选，若通道独立主线表现不足再纳入 |
| foundation model | Timer / TimesFM / Chronos / Moirai 等 | 成本较高，建议后置到 1.4c 或独立消融 |

Stage 1.4b 的第一批不宜过大。建议先选：

- DLinear 或 NLinear；
- FITS 或一个频域专家；
- PatchTST；
- iTransformer 或 TSMixer/TimeMixer。

如果 Quito 官方 benchmark runner 已支持某些模型，应优先选择这些模型，减少数据口径和复现成本。

### 4.3 Stage 1.4c：专家筛选与最终池固化

目的：

> 根据真实预测缓存和 cell-level profiling 结果，筛选最终 3-4 个专家，供 Stage 1.5 gate 使用。

筛选不按平均 MSE 单独决定，而看互补性：

- `cell_wise_rank_diversity`：不同 TSF cell 中专家排序是否变化。
- `cell_wise_win_rate`：每个专家在哪些 cell 中胜出。
- `oracle_contribution`：加入该专家后 oracle ensemble 是否明显提高。
- `error_correlation`：两个专家误差高度相关时只保留更快或更稳的。
- `latency_ms_per_window`：在线或批量推理成本。
- `failure_rate`：训练失败、NaN、异常预测比例。
- `few_shot_adaptation_cost`：如果后续 few-shot 微调，单位收益成本。

如果 oracle ensemble 上界不明显高于 best fixed expert，优先重选专家池，而不是直接训练 router。

## 5. 预测缓存 schema

第一版建议输出目录：

```text
outputs/vision_ts_routing/expert_predictions/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/
    <expert_set_id>/
      predictions.parquet
      errors.parquet
      manifest.json
      profiling/
        cell_model_matrix.csv
        oracle_summary.csv
```

`predictions.parquet` 建议列：

| 列名 | 含义 |
| --- | --- |
| `physical_window_id` | 主键 |
| `sample_set_id` | 采样集合 |
| `base_registry_id` | 合法窗口母体 |
| `split` | train / valid / test |
| `subset` | hour / min |
| `item_id` | QuitoBench item |
| `channel` | sample-channel 通道 |
| `history_start_idx` | history 起点 |
| `target_start_idx` | target 起点 |
| `expert_id` | 专家唯一 ID |
| `expert_family` | 专家族 |
| `pred_len` | 预测长度 |
| `prediction` | 长度为 `pred_len` 的数组列，或拆成 `yhat_0...yhat_95` |
| `prediction_format` | `array` 或 `wide_columns` |

`errors.parquet` 建议列：

| 列名 | 含义 |
| --- | --- |
| `physical_window_id` | 主键 |
| `sample_set_id` | 采样集合 |
| `expert_id` | 专家 ID |
| `mse` | 当前窗口 MSE |
| `mae` | 当前窗口 MAE |
| `smape` | 可选 |
| `rank_in_window` | 当前窗口内专家排名 |
| `is_oracle_top1` | 是否为窗口最优专家 |
| `soft_oracle_weight` | 后续 gate 可用的 soft target，Stage 1.4 可先生成 |

`manifest.json` 必须记录：

- `expert_set_id`
- `sample_set_id`
- `base_registry_id`
- `registry_path`
- `expert_ids`
- 每个专家的来源框架、版本、配置、训练 split、device、seed
- 是否使用 future：必须为 `false`
- target 使用范围：仅 error/oracle 计算
- 输出行数、窗口覆盖率、NaN 比例
- latency 统计

数组列与 wide columns 的选择：

- 第一版建议优先使用 wide columns：`yhat_0 ... yhat_95`，便于 parquet、pandas、polars 和后续误差矩阵处理。
- 如果某个框架输出天然是 ndarray，可在中间层使用 array，正式缓存仍转 wide columns。

## 6. 与路线 1 / 路线 2 的关系

Stage 1.4 是共享基础设施，不单独属于路线 1 或路线 2。

路线 1 使用方式：

- 不使用 TSF cell 标签训练 gate。
- 使用专家预测缓存生成 oracle error / soft oracle。
- 评估 visual embedding + proxy 是否比 fixed ensemble、best fixed expert、statistical router 更接近 oracle。
- 重点报告整体性能、oracle gap、expert utilization、latency。

路线 2 使用方式：

- 使用官方 TSF cell 做专家画像和解释。
- 分析不同 cell 中的 expert preference。
- 后续可用 `L_tsf + L_route` 做视觉结构监督或 few-shot adaptation。
- 重点报告 cell-level rank、win rate、oracle contribution、leave-one-cell-out 泛化。

因此 Stage 1.4 的缓存 schema 必须同时支持：

- 按窗口训练路线 1 gate；
- 按 official TSF cell 聚合路线 2 profiling；
- 后续复用到不同 `sample_set_id` 或外部数据集。

## 7. 参考先验

Stage 1.4 借鉴以下方向，但不绑定其完整实验设置：

- TimeFuse：样本级模型融合和 meta-feature routing 说明不同预测模型在不同样本上存在互补性，适合作为统计 router 与 sample-level fusion 对照。
- TimeRecipe：模块级 benchmarking 思路可迁移为专家池消融，即 decomposition、frequency、patch、transformer、MLP/multiscale 等能力族覆盖。
- QuitoBench：官方 TSF regime/cell 分层适合作为专家画像和路线 2 结构解释主轴。
- Time-Series-Library/tslib：优先复用其常见模型实现和训练脚本，减少手写模型风险。

参考链接：

- TimeFuse: https://arxiv.org/abs/2505.18442
- TimeRecipe: https://openreview.net/forum?id=CsoR8ztROC
- QuitoBench: https://arxiv.org/abs/2603.26017
- Quito GitHub: https://github.com/alipay/quito
- Time-Series-Library: https://github.com/thuml/Time-Series-Library

## 8. 验证要求

Stage 1.4a 最小验证：

1. `predictions.parquet` 中 `(physical_window_id, expert_id)` 唯一。
2. 覆盖的 `physical_window_id` 均来自 Stage 1.0 registry。
3. 每个 expert 的预测长度等于 `pred_len=96`。
4. 不读取 future 作为输入；target 只用于 error 计算。
5. 所有误差列无 NaN / inf。
6. `sample_set_id` 保留且与 registry 一致。
7. 输出 manifest 记录专家配置和覆盖率。
8. profiling 能输出 `cell x expert` 性能矩阵。

Stage 1.4b 增加验证：

1. 每个复用框架专家都能单独复现实验命令。
2. 训练、验证、测试 split 口径与 Quito registry 对齐。
3. 每个专家记录训练耗时、推理耗时和失败率。
4. 至少输出一个 oracle summary，比较 oracle ensemble、best fixed expert、uniform ensemble。
5. 若复用上游 runner，必须验证上游样本顺序能映射回 `physical_window_id`。

## 9. 推荐下一步

1. 编写 Stage 1.4a 实现计划，先只覆盖极轻量专家和 cache schema。
2. 审计 Quito repo 和 Time-Series-Library/tslib 的可复用 runner，记录最小接入路径。
3. 实现 `expert_predictions` 缓存写入和误差计算，不接 router。
4. 在 smoke sample 上验证 schema 后，再扩展到 Stage 1.0 working registry。
5. 并行推进 Stage 1.3a visual encoder adapter smoke，但二者通过 `physical_window_id` 和后续 oracle error 对齐，不互相阻塞。

