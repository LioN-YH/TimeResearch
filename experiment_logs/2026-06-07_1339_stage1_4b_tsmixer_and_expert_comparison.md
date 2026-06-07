# Stage 1.4b：TSMixer smoke 与四专家同样本互补性汇总

## 1. 目的

在 DLinear 和 PatchTST smoke 后，接入第三个训练型专家 TSMixer，并在同一 5k stratified sample 上汇总 seasonal naive、DLinear、PatchTST、TSMixer 的 oracle gap。

本实验不实现 router，不运行视觉 encoder，不修改 Quito 上游代码。

## 2. 实现范围

本次整理了训练型专家 runner：

- 新增通用入口：`tools/quitobench_framework_expert_cache.py`
- 保留兼容入口：`tools/quitobench_dlinear_expert_cache.py`

兼容入口用于保证历史日志中的 DLinear 命令仍可运行；新实验优先使用通用入口。

新增比较脚本：

- `tools/quitobench_expert_cache_comparison.py`

该脚本只读取多个 expert cache 的 `errors.parquet`，按共同 `physical_window_id` 汇总：

- best fixed expert
- uniform MSE proxy
- oracle MSE
- oracle gap vs best fixed
- expert oracle top1 rate
- split / cell 层级汇总

## 3. TSMixer smoke 命令

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model tsmixer \
  --stratified-rows 5000 \
  --epochs 1 \
  --batch-size 128 \
  --expert-set-id tsmixer_v1__stratified_smoke_5k_cuda \
  --device cuda \
  --num-blocks 2 \
  --d-ff 64 \
  --norm-type layer
```

## 4. TSMixer 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_smoke_5k_cuda/
```

包含：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

## 5. TSMixer 校验结果

```text
stage stage1_4b_tsmixer_expert_cache_smoke
expert_ids ['tsmixer_quito']
source_model quito.models.tsmixer.TSMixer
windows 5000
prediction_rows 5000
error_rows 5000
prediction_unique True
error_unique True
splits {'train': 1695, 'valid': 1667, 'test': 1638}
cells 8
train_stats {'train_windows': 1695, 'trained_splits': ['train'], 'epochs_completed': 1, 'final_train_loss': 1.4860554933547974, 'train_elapsed_seconds': 0.5522097134962678, 'device': 'cuda:0'}
soft_weight_max_abs_error 0.0
implements_router False
runs_visual_encoder False
runs_neural_experts True
```

## 6. 四专家 comparison 命令

```bash
conda run -n quito python tools/quitobench_expert_cache_comparison.py \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_smoke_5k_cuda \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_smoke_5k_cuda \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_smoke_5k_cuda \
  --required-experts seasonal_naive,dlinear_quito,patchtst_quito,tsmixer_quito \
  --comparison-id seasonal_naive_dlinear_patchtst_tsmixer__stratified_smoke_5k
```

## 7. comparison 输出

```text
outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/seasonal_naive_dlinear_patchtst_tsmixer__stratified_smoke_5k/
```

包含：

- `comparison_summary.csv`
- `comparison_by_split.csv`
- `comparison_by_cell.csv`
- `expert_metrics.csv`
- `manifest.json`

## 8. comparison 结果

整体汇总：

```text
num_common_windows 5000
num_experts 4
oracle_mse 22472246402.179615
best_fixed_expert seasonal_naive
best_fixed_mse 28504308765.4743
uniform_mse_proxy 226854616508.8491
oracle_gap_vs_best_fixed 6032062363.294685
```

专家均值与 oracle top1：

```text
seasonal_naive mse=28504308765.4743 oracle_top1_rate=0.7104
patchtst_quito mse=164175105747.9826 oracle_top1_rate=0.1770
dlinear_quito mse=323707825833.5170 oracle_top1_rate=0.0786
tsmixer_quito mse=391031225688.4227 oracle_top1_rate=0.0340
```

split 层级：

```text
test  oracle_gap_vs_best_fixed=3752694148.3948517
train oracle_gap_vs_best_fixed=10450321993.027431
valid oracle_gap_vs_best_fixed=3779305952.7421618
```

cell 层级中 `highT_lowS_lowF` 的 best fixed expert 为 `dlinear_quito`，其余 cell 当前 best fixed 多为 `seasonal_naive`。

## 9. 结论

TSMixer smoke 跑通，但在当前 5k stratified、单 epoch smoke 设置下，TSMixer 的平均误差和 oracle top1 rate 均弱于 seasonal naive、PatchTST 和 DLinear。

四专家 comparison 说明：

- seasonal naive 是当前同样本上的整体 best fixed expert；
- PatchTST 仍有明显窗口级 top1 贡献；
- DLinear 在少数 cell 上有补充价值，尤其 `highT_lowS_lowF`；
- 加入 TSMixer 后 oracle MSE 只比三专家略低，当前 smoke 下边际贡献较小；
- oracle gap 仍存在，说明后续 router/gate 有学习空间，但不能直接把当前 single-epoch smoke 当作最终专家画像。

## 10. 下一步

1. 先不要全量固化 DLinear/PatchTST/TSMixer；当前训练轮数和 OOF 口径都还不是正式设置。
2. 下一步更值得做的是对同一 5k sample 调整训练预算或归一化策略，确认 neural experts 是否因单 epoch 不足而低估。
3. 在进入 router 前，规划 train split 的 out-of-fold expert prediction，避免 train-in error 过于乐观。
