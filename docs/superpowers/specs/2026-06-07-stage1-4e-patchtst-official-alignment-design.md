# Stage 1.4e PatchTST 官方口径对齐与尺度审计设计

## 1. 背景

Stage 1.4c/1.4d 中，DLinear 在 20k/50k 分层样本上表现出稳定互补性，但 PatchTST 在当前 runner 下出现异常大的预测误差，TSMixer 也未形成稳定收益。这个结果与 QuitoBench 论文和官方配置中 PatchTST 通常强于 DLinear 的经验不一致。

当前不能直接得出“PatchTST 不适合作为专家”的结论。已确认的主要口径差异包括：

- 当前任务为 `seq_len=192, pred_len=96`，不是 QuitoBench 官方示例网格中的 `96/48/S` 等任务；
- 当前 wrapper 直接使用 raw window 训练和推理，只依赖模型内部 per-window RevIN；
- Quito `TimeSeriesDataset.process_raw_df()` 会按 train 段计算 `mean/std` 并对整段序列标准化；
- 官方 PatchTST 训练配置约为 `learning_rate=1e-4`、`num_epochs=20`、`batch_size=128`、`drop_last=true`、cosine scheduler；
- 当前 Stage 1.4c/1.4d 使用 `learning_rate=1e-3/3e-4`、`epochs=1/5`、`batch_size=32`、无 scheduler。

Stage 1.4e 的目标是先审计实现与实验口径，再决定是否扩大训练预算或调整专家池。

## 2. 目标

Stage 1.4e 需要回答四个问题：

1. 当前 PatchTST 异常是否来自 wrapper 的输入尺度、输出尺度或 RevIN 路径错误；
2. 在更接近 QuitoBench 官方配置的 `96/48/S` 任务上，PatchTST 是否恢复到合理表现；
3. 在当前项目任务 `192/96/S` 上，官方对齐训练配置是否仍然使 PatchTST 明显弱于 DLinear/seasonal naive；
4. 后续是否应继续保留 PatchTST 作为 OOF/router 候选专家，还是先转向其他模型池。

## 3. 非目标

本阶段不实现 router/gate，不运行视觉 encoder，不生成 OOF cache，不改 Quito 上游模型代码，不把 PatchTST 结果直接扩展到全量 registry。

本阶段不以刷榜为目标，只做可解释的实现审计和小规模官方口径 sanity。

## 4. 推荐方案

采用两段式方案：先做最小审计，再做官方口径 sanity。

第一段是代码与数据尺度审计：

- 构造 toy batch，验证 DLinear/PatchTST 的 `loss()` 与 `predict()` 是否在 RevIN 打开时使用一致尺度；
- 对已有 PatchTST cache 做 prediction 分布审计，统计 `yhat`、target、absolute error 的 min/max/mean/std 和关键分位数；
- 显式检查 NaN/Inf、极端预测、loss 与 prediction MSE 是否存在尺度不一致；
- 记录审计结果到 Stage 1.4e 实验日志。

第二段是官方口径训练 sanity：

- 在 runner 中增加可选 train-set scaler，按 train split 的同一 item/channel 历史和目标可见序列估计 mean/std；
- 训练和推理内部使用标准化数据；
- 写入 prediction cache 前 inverse transform 回原尺度，保持 Stage 1.4 cache schema 不变；
- 增加 `drop_last`、cosine scheduler、`num_workers`、`eval_batch_size` 等显式参数；
- 使用 `lr=1e-4`、`epochs=20`、`batch_size=128`、`drop_last=true`、cosine scheduler 跑 DLinear/PatchTST sanity 矩阵。

## 5. 实验矩阵

最小推荐矩阵如下：

| task | model | scaler | lr | epochs | batch | scheduler | 目的 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `96/48/S` | DLinear | train-set | `1e-4` | 20 | 128 | cosine | 官方对齐 DLinear 参考 |
| `96/48/S` | PatchTST | train-set | `1e-4` | 20 | 128 | cosine | 验证 PatchTST 是否恢复正常 |
| `192/96/S` | DLinear | train-set | `1e-4` | 20 | 128 | cosine | 当前任务官方训练口径参考 |
| `192/96/S` | PatchTST | train-set | `1e-4` | 20 | 128 | cosine | 判断当前任务是否对 PatchTST 不友好 |

如 GPU 资源允许，四组实验可用 `CUDA_VISIBLE_DEVICES=0/1/2/3` 分开跑。runner 内部仍只看到 `cuda:0`，manifest 必须记录外部 GPU 分配和完整命令。

## 6. 判定标准

如果 `96/48/S` 下 PatchTST 恢复到合理 MSE，并明显接近或优于 DLinear，则 Stage 1.4c/1.4d 的 PatchTST 异常主要来自实验口径不对齐，不能用于否定 PatchTST。

如果 `96/48/S` 正常，但 `192/96/S` 仍显著弱于 DLinear/seasonal naive，则当前项目任务、样本选择或跨 cell 分布可能对 PatchTST 更困难。后续应按 cell/subset 分析，而不是整体否定。

如果两种任务下 PatchTST 都异常，但尺度审计通过，则需要继续检查 sample-channel wrapper 与 Quito 官方 dataset 切窗/训练 loop 差异。

如果尺度审计发现 prediction inverse transform、RevIN 口径或 cache 写入异常，则先修 wrapper，再重跑对比；此前 PatchTST 结果标记为不可用于模型选择。

## 7. 代码边界

优先修改：

- `tools/quitobench_framework_expert_cache.py`
- `tests/test_quitobench_dlinear_expert_cache.py`

必要时新增独立诊断工具：

- `tools/quitobench_expert_prediction_diagnostics.py`

不修改：

- `quito/quito/models/*.py`
- `quito/quito/datasets.py`
- Stage 1.2 imageization 和视觉 encoder 相关代码
- router/gate/OOF 相关代码

## 8. 产物

Stage 1.4e 完成后应产生：

- 新的 runner 参数和对应测试；
- PatchTST/DLinear toy scale 审计结果；
- 训练 sanity 矩阵的 prediction/error cache；
- comparison summary；
- `experiment_logs/2026-06-07_HHMM_stage1_4e_patchtst_official_alignment.md`；
- `experiment_logs/实验日志总览.md` 中新增 Stage 1.4e 记录。

## 9. 风险

train-set scaler 的精确口径需要谨慎。Quito 官方 dataset 是在完整 item/channel 序列上按 train 段估计 `mean/std`，然后切 train/valid/test window；当前 registry wrapper 只有窗口级 history/target 抽取接口。如果无法可靠还原完整序列级 scaler，必须在 manifest 中明确标注为 wrapper-level approximation，并避免声称完全复现官方训练。

官方 `96/48/S` 任务可能需要新 registry 或临时 registry。若生成成本较高，应先用小样本 smoke 验证参数链路，再启动 20k/50k sanity。

PatchTST 训练时间可能增加。正式实验必须记录 GPU、batch size、train window 数、elapsed seconds、final train loss、valid/test MSE 和 prediction 分布，避免只报告单一 MSE。
