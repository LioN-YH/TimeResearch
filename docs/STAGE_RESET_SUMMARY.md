# Stage Reset Summary

本文只保留目前做对、有效、后续值得继承的部分，同时明确需要停止关注或谨慎使用的路线。目标是把项目从 Stage 1.4 之后的混乱推进中收敛回来。

## 1. 当前真实目标

项目目标不是让视觉模型直接做时序预测，也不是让 ViT 记住某个窗口设置下的专家标签，而是：

> 从通道独立的 sample-channel 历史窗口中学习跨窗口长度、预测 horizon 和数据源都尽量稳定的时序结构先验，并用该先验驱动多个异构时序专家的连续自适应融合。

核心对象要分清：

| 对象 | 粒度 | 用途 |
| --- | --- | --- |
| official cluster / TSF cell | item 级、多通道、全长 | 结构解释、分层报告、路线 2 辅助监督 |
| channel full-length STL | item-channel 级、全长 | 通道异质性和 cluster 解释辅助 |
| sample-channel window | 单窗口、单通道、history-only | 视觉模块和 gate 的实际输入 |
| expert error / soft oracle | sample-channel + horizon | 路线 1 router/gate 主监督 |

## 2. 双路线方案是否保留

保留，不需要大改。

原因：

1. QuitoBench 已经能提供足够多的 item-channel windows，足以支持 sample-level 专家融合验证。
2. 官方 cluster/codebook 已被定位和反推，可支持结构分层和解释线。
3. Stage 1 已经形成共享工程资产：window registry、light proxy、imageization、expert cache schema。
4. 视觉先验作为下游 continuous gate 的输入，仍然比 hard cluster router 更符合“跨窗口/horizon/数据源稳定结构”的目标。

需要调整的是推进方式，而不是大方向：

- 路线 1 先固定一个干净任务口径，建立专家预测缓存和 gate 评估闭环。
- 路线 2 只在路线 1 的共享缓存和 embedding 稳定后做分层解释、few-shot 和跨 cell 泛化。
- 暂时不要继续扩展 Stage 1.4 之后的并行实验分支。

## 3. Quitobench 已确认关键事实

已确认：

- QuitoBench benchmark 包含 `hour` 和 `min` 两个 subset，共 1,290 个 item、6,450 条 item-channel 序列。
- 官方数据最新公开 parquet 删除了 `cluster`，但 HF revision `17362dcb` 保留 `cluster` 列。
- `cluster` 是 item 级常量，README schema 标注为 8-class TSF regime label。
- Stage 0.6b 反推 codebook 高置信：

| code | cell |
| ---: | --- |
| 0 | `highT_highS_highF` |
| 2 | `highT_highS_lowF` |
| 6 | `highT_lowS_highF` |
| 8 | `highT_lowS_lowF` |
| 18 | `lowT_highS_highF` |
| 20 | `lowT_highS_lowF` |
| 24 | `lowT_lowS_highF` |
| 26 | `lowT_lowS_lowF` |

- Stage 0.7 通道级 full-length STL 对官方 codebook 的 item exact match 为 68.37%，forecastability 维度 100%，trend/seasonality 不完全一致。
- 因此官方 cluster 是路线 2 主标签；STL/proxy 是解释辅助，不能替代官方标签。

## 4. Stage0/Stage1 值得继承的成果

Stage 0 值得继承：

- 数据位置、schema、下载和 sufficiency audit。
- item/channel 数、窗口规模和 8 cell 覆盖结论。
- 官方 cluster 来源定位和 `quitobench_tsf_cells_final.csv`。
- Stage 0.6b 官方 codebook 反推。
- Stage 0.7 通道级 STL 中间结果，用于通道异质性和伪图像解释。

Stage 1 值得继承：

- `physical_window_id / base_registry_id / sample_set_id` 三层 ID schema。
- `split_context_policy=quito_overlap`，对齐 Quito valid/test history overlap。
- sample-channel light proxy 离线 cache 和 torch online kernel。
- 三视图 imageization：`line_raster`、`period_fold`、`fft_power`。
- visual embedding cache smoke 的 `physical_window_id` 对齐方式。
- expert cache schema：`predictions.parquet`、`errors.parquet`、profiling、manifest。
- SNaive、DLinear、PatchTST 的初步 wrapper，尤其 Stage 1.4f 之后的 Quito train-set scaler 复用。

## 5. 不建议继续关注的路线

不建议继续沿用：

- Stage 0 早期 full channel/item STL timeout 路径：已被轻量 proxy、item STL 和后续 channel STL 替代。
- 旧 `cfcd86e70e73` strict/coarse registry：未对齐 Quito overlap，不作为主输入。
- Stage 1.4e 以前 raw-scale 神经专家结论：标准化口径错误或不完整。
- Stage 1.4e wrapper-level global window standardizer：只作过渡诊断，不作为主口径。
- Stage 1.4g-b raw sparse 结果中“神经模型远弱于 SNaive”的结论：后续发现 period、normalize、超参和 all-channel 口径未对齐。
- 继续盲目加入 TSMixer/更多专家/更多大训练：当前问题是口径收敛和缓存可信度，不是专家数量不足。

需要降级为“不确定”的结果：

- PatchTST 是否稳定优于 SNaive：在 ind_1 normalized sparse 中表现好，在 all-channel raw-scale MAE 上仍不稳；需要官方 full dense 或更严格 sparse OOF 证据。
- sparse stride=96/288 是否代表 full dense：目前只能作为交互式 sanity，不是官方复现。
- visual embedding latency：Stage 1.3a0 后一次复测受 I/O/GPU 占用影响，不作为性能结论。

## 6. 当前代码和实验流程风险

数据加载和 split：

- Stage 1.0 后 `quito_overlap` 对齐 Quito，是正确口径。
- valid/test history overlap 不算 target 泄漏；但文档和 manifest 必须明确 target 仍在当前 split。
- sparse registry 的 stride 是降采样策略，不等于 Quito dense rolling window。

降采样：

- `sample_stride=96/288` 可以做工程 sanity，但不能默认统计代表性充分。
- `split/subset/official_tsf_cell` 分层抽样是合理起点，但还缺少 item/channel 均衡控制。
- 后续 gate 训练建议固定一个 canonical sparse set，并保留一个 untouched sparse test set。

标准化：

- Quito `TimeSeriesDataset` 用 train 段 item/channel scaler，这是后续主口径。
- 当前 Quito 源码即使 `normalize=False` 也会标准化；不要依赖该开关。
- Stage 1.4f 后的 `--train-set-standardize` 方向正确。
- 比较专家时必须说明 raw-scale MSE/MAE 还是 normalized-scale MSE/MAE。

专家缓存：

- 训练型专家 wrapper 已限制只用 train split 训练，这是正确的。
- `target` 只用于 loss/error/oracle，预测只读 history。
- comparison 脚本的 `uniform_mse_proxy` 是专家 MSE 均值，不是真正 uniform prediction ensemble。
- 已新增 common-window oracle target audit；当前三专家 50k cache 的交集为 23,456 windows，raw-scale best fixed 为 DLinear，oracle gap vs best fixed 约 `1.87e10`，可支持低成本 gate smoke。
- 已完成 clean `matrix50k_v1` 三专家 cache rebuild：SNaive / DLinear / PatchTST 复用同一固定 sampled registry，cache audit 确认 common prediction/error windows 均为 `50000`。
- clean raw-scale oracle audit 中 best fixed 为 DLinear，oracle gap vs best fixed 约 `4.29e10`；但 clean normalized-scale MSE best fixed 变为 SNaive，且 DLinear/PatchTST normalized MSE 被少数大误差显著放大。
- PatchTST 在当前 raw-scale common-window audit 中 top1 rate 只有约 6%，暂不能作为“稳定强专家”结论。
- soft oracle 不应直接基于当前 raw MSE 或未排查的 normalized MSE 进入 gate；下一步必须先做 normalized error 分布、small-std/outlier 和 subset/cell 分解审计。

## 7. 后续最小可行推进路线

建议先收敛到一个 canonical 实验：

```text
dataset: QuitoBench revision 17362dcb
features: S, all channels
history_len: 96
pred_len: 48
split_context_policy: quito_overlap
sampling: fixed sparse registry, stratified by split/subset/official_tsf_cell, with item/channel balance
standardization: Quito train segment item/channel scaler
experts: SNaive(period=6), DLinear, PatchTST
metrics: raw-scale MSE/MAE + normalized-scale MSE/MAE, both sample-weighted
```

先只做：

1. 固化 canonical registry 和 manifest。
2. 固化三专家 prediction/error cache。当前 clean `matrix50k_v1` 已完成，但 normalized-scale 异常仍需排查。
3. 生成真实 oracle、best fixed、真实 uniform prediction ensemble 和 soft oracle target。raw/normalized audit 已完成，soft oracle target 暂缓。
4. 排查 normalized MSE pathology，确认是否由 small std、outlier、subset/cell 或训练预算导致。
5. 排查通过后再跑 visual embedding cache 和最小 gate baseline：proxy-only、visual-only、proxy+visual。

暂时不做：

- 大规模 full dense finetune。
- 新专家大扩展。
- 端到端专家联合训练。
- 把 official cluster 当 router 硬标签。

## 8. 下一阶段建议实验设计

下一阶段建议按证据链推进：

1. Canonical data audit：clean `matrix50k_v1` 已有 registry audit；后续补 item/channel 均衡和 window std 分布。
2. Normalized error pathology audit：基于 clean `matrix50k_v1` 分解 DLinear/PatchTST 的 normalized MSE outlier，重点看 window std、subset/cell、item/channel。
3. Expert metric policy audit：补充 winsorized MSE、MAE/Huber 或官方 `evaluate_series` 口径，判断 gate target 应使用哪个误差定义。
4. Common-window visual embedding cache：只有 normalized pathology 解释清楚后，再基于 clean 50k `physical_window_id` 生成三视图 visual embedding。
5. Gate smoke：用 clean 50k 做 proxy-only / visual-only / proxy+visual 的最小验证，明确 raw/normalized target 口径。

判定标准：

- 若 proxy+visual 相比 proxy-only 有稳定收益，继续路线 1 和路线 2。
- 若 visual-only 不强但 proxy+visual 有收益，视觉先验仍可作为互补信号。
- 若视觉完全无收益，先审视 imageization/encoder，而不是继续加专家。
- 若专家 oracle gap 很小，说明专家池互补不足，应优先换专家池而不是训练 gate。
