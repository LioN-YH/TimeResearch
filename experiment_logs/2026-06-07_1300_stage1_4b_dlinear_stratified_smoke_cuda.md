# Stage 1.4b：DLinear 分层抽样 GPU smoke

## 1. 目的

在 `dlinear_v1__smoke` 前缀截断 smoke 之后，使用 `split/subset/official_tsf_cell` 分层抽样重新验证 DLinear 训练型专家缓存。

本实验继续只接入 DLinear，不实现 router，不运行视觉 encoder，不修改 Quito 上游代码。专家训练只使用 `train` split；预测缓存覆盖 `train/valid/test`，供后续 router/gate 训练、验证和最终测试分别使用。

## 2. 分层抽样含义

分层抽样不是 Stage 1.4a-expanded 分析。它是 Stage 1.4b DLinear smoke 的取样方式修正：

- 不再简单取 registry 前 N 行；
- 按 `split/subset/official_tsf_cell` 的非空组合均衡抽取窗口；
- 避免小样本 smoke 被 registry 排序、单个 item 或单个 TSF cell 偏置；
- 输出仍复用 Stage 1.4a 的 `predictions/errors/manifest/profiling` schema。

## 3. 输入

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- expert_set_id: `dlinear_v1__stratified_smoke_5k_cuda`
- stratified_rows: `5000`
- stratify_cols: `split,subset,official_tsf_cell`
- epochs: `1`
- batch_size: `128`
- device: `cuda`

## 4. 命令

```bash
conda run -n quito python tools/quitobench_dlinear_expert_cache.py \
  --stratified-rows 5000 \
  --epochs 1 \
  --batch-size 128 \
  --expert-set-id dlinear_v1__stratified_smoke_5k_cuda \
  --device cuda
```

校验：

```bash
conda run -n quito python -c "import json; from pathlib import Path; import pandas as pd; out=Path('outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_smoke_5k_cuda'); manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')); pred=pd.read_parquet(out/'predictions.parquet'); err=pd.read_parquet(out/'errors.parquet'); cell=pd.read_csv(out/'profiling/cell_model_matrix.csv'); print('windows', manifest['total_windows']); print('prediction_rows', len(pred)); print('error_rows', len(err)); print('prediction_unique', pred[['physical_window_id','expert_id']].duplicated().sum() == 0); print('error_unique', err[['physical_window_id','expert_id']].duplicated().sum() == 0); print('splits', pred['split'].value_counts().to_dict()); print('subsets', pred['subset'].value_counts().to_dict()); print('cells', pred['official_tsf_cell'].nunique()); print('cell_matrix_rows', len(cell)); print('train_stats', manifest['training_stats']); print('sampling', manifest['sampling_summary']); print('soft_weight_max_abs_error', float((err.groupby('physical_window_id')['soft_oracle_weight'].sum() - 1.0).abs().max())); print('implements_router', manifest['implements_router']); print('runs_visual_encoder', manifest['runs_visual_encoder']); print('runs_neural_experts', manifest['runs_neural_experts'])"
```

## 5. 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_smoke_5k_cuda/
```

包含：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

## 6. 验证结果

执行输出：

```text
[done] output_dir=/home/user10/TSF/DATAPrepare/.worktrees/stage1-4b-framework/outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_smoke_5k_cuda
[done] windows=5000
[done] prediction_rows=5000
[done] train_windows=1695
```

校验输出：

```text
windows 5000
prediction_rows 5000
error_rows 5000
prediction_unique True
error_unique True
splits {'train': 1695, 'valid': 1667, 'test': 1638}
subsets {'min': 2929, 'hour': 2071}
cells 8
cell_matrix_rows 8
train_stats {'train_windows': 1695, 'trained_splits': ['train'], 'epochs_completed': 1, 'final_train_loss': 27931034.0, 'train_elapsed_seconds': 0.7990673752501607, 'device': 'cuda:0'}
sampling {'strategy': 'stratified', 'requested_rows': 5000, 'selected_rows': 5000, 'group_cols': ['split', 'subset', 'official_tsf_cell'], 'group_count': 36}
soft_weight_max_abs_error 0.0
implements_router False
runs_visual_encoder False
runs_neural_experts True
```

## 7. 执行说明

用户指出首次分层 5k 命令不应使用 CPU。该 CPU 命令被中断，虽然输出目录可能已经写出文件，但不作为本实验正式结论。本日志只记录 `dlinear_v1__stratified_smoke_5k_cuda` GPU 版结果。

当前 DLinear smoke 未做尺度归一化策略消融，`final_train_loss` 只用于确认训练 loop 跑通，不作为模型效果结论。

## 8. 结论

Stage 1.4b DLinear 分层抽样 GPU smoke 跑通。

当前已经验证：

- 分层抽样能覆盖 `train/valid/test`、`hour/min` 和 8 个官方 TSF cell；
- 专家训练只使用 `train` split；
- `train/valid/test` 均生成预测缓存和误差缓存；
- 预测能映射回 `physical_window_id`；
- 输出复用 Stage 1.4a 的 `predictions/errors/manifest/profiling` schema；
- 未实现 router；
- 未运行视觉 encoder。

## 9. 下一步

1. 将 DLinear 分层 smoke 的抽样策略固化为后续训练型专家 smoke 默认策略。
2. 接入 NLinear，复用同一 `RegistryWindowDataset`、分层抽样和 cache 写出逻辑。
3. 后续若进入 router 训练，`train` split 专家误差建议升级为 out-of-fold prediction，降低 train-in prediction 对 oracle target 的乐观偏差。
