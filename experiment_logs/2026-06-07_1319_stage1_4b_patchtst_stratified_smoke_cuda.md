# Stage 1.4b：PatchTST 分层抽样 GPU smoke

## 1. 目的

在 DLinear 训练型专家 smoke 跑通后，接入结构差异更明显的 PatchTST 专家，继续验证 Stage 1.4 预测缓存协议。

本实验不实现 router，不运行视觉 encoder，不修改 Quito 上游代码。专家训练只使用 `train` split；预测缓存覆盖 `train/valid/test`。

## 2. 背景决策

Stage 1.4a-expanded 已完成 seasonal naive 轻量专家 baseline 的全量固化，后续专家互补性分析可以复用。

NLinear 暂不作为当前优先项：

- Quito 本地没有 NLinear 实现或配置；
- 本地 Time-Series-Library 也未发现直接可用的 `NLinear.py`；
- NLinear 与 DLinear 同属线性专家族，互补性弱于 PatchTST。

因此本实验优先接入 PatchTST。后续第三个训练型专家建议考虑 TSMixer。

## 3. 输入

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- expert_set_id: `patchtst_v1__stratified_smoke_5k_cuda`
- stratified_rows: `5000`
- stratify_cols: `split,subset,official_tsf_cell`
- epochs: `1`
- batch_size: `128`
- device: `cuda`

PatchTST smoke 配置：

- `seq_len=192`
- `pred_len=96`
- `patch_len=16`
- `stride=8`
- `d_model=128`
- `d_ff=256`
- `n_heads=4`
- `e_layers=2`
- `revin=True`

## 4. 命令

```bash
conda run -n quito python tools/quitobench_dlinear_expert_cache.py \
  --expert-model patchtst \
  --stratified-rows 5000 \
  --epochs 1 \
  --batch-size 128 \
  --expert-set-id patchtst_v1__stratified_smoke_5k_cuda \
  --device cuda \
  --d-model 128 \
  --d-ff 256 \
  --n-heads 4 \
  --e-layers 2 \
  --patch-len 16 \
  --stride 8
```

校验：

```bash
conda run -n quito python -c "import json; from pathlib import Path; import pandas as pd; out=Path('outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_smoke_5k_cuda'); manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')); pred=pd.read_parquet(out/'predictions.parquet'); err=pd.read_parquet(out/'errors.parquet'); cell=pd.read_csv(out/'profiling/cell_model_matrix.csv'); print('stage', manifest['stage']); print('expert_ids', manifest['expert_ids']); print('source_model', manifest['source_model']); print('windows', manifest['total_windows']); print('prediction_rows', len(pred)); print('error_rows', len(err)); print('prediction_unique', pred[['physical_window_id','expert_id']].duplicated().sum() == 0); print('error_unique', err[['physical_window_id','expert_id']].duplicated().sum() == 0); print('splits', pred['split'].value_counts().to_dict()); print('subsets', pred['subset'].value_counts().to_dict()); print('cells', pred['official_tsf_cell'].nunique()); print('cell_matrix_rows', len(cell)); print('train_stats', manifest['training_stats']); print('sampling', manifest['sampling_summary']); print('soft_weight_max_abs_error', float((err.groupby('physical_window_id')['soft_oracle_weight'].sum() - 1.0).abs().max())); print('implements_router', manifest['implements_router']); print('runs_visual_encoder', manifest['runs_visual_encoder']); print('runs_neural_experts', manifest['runs_neural_experts'])"
```

## 5. 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_smoke_5k_cuda/
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
[done] output_dir=/home/user10/TSF/DATAPrepare/.worktrees/stage1-4b-framework/outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_smoke_5k_cuda
[done] windows=5000
[done] prediction_rows=5000
[done] train_windows=1695
```

校验输出：

```text
stage stage1_4b_patchtst_expert_cache_smoke
expert_ids ['patchtst_quito']
source_model quito.models.patchtst.PatchTST
windows 5000
prediction_rows 5000
error_rows 5000
prediction_unique True
error_unique True
splits {'train': 1695, 'valid': 1667, 'test': 1638}
subsets {'min': 2929, 'hour': 2071}
cells 8
cell_matrix_rows 8
train_stats {'train_windows': 1695, 'trained_splits': ['train'], 'epochs_completed': 1, 'final_train_loss': 1.428146243095398, 'train_elapsed_seconds': 0.6482760412618518, 'device': 'cuda:0'}
sampling {'strategy': 'stratified', 'requested_rows': 5000, 'selected_rows': 5000, 'group_cols': ['split', 'subset', 'official_tsf_cell'], 'group_count': 36}
soft_weight_max_abs_error 0.0
implements_router False
runs_visual_encoder False
runs_neural_experts True
```

## 7. 结论

Stage 1.4b PatchTST 分层抽样 GPU smoke 跑通。

当前已经验证：

- Quito PatchTST 可以复用当前训练型专家缓存 runner；
- 专家训练只使用 `train` split；
- `train/valid/test` 均生成预测缓存和误差缓存；
- 预测能映射回 `physical_window_id`；
- 输出复用 Stage 1.4a 的 `predictions/errors/manifest/profiling` schema；
- 未实现 router；
- 未运行视觉 encoder。

## 8. 下一步

1. 将当前脚本从 DLinear 专名整理为训练型专家通用 runner，保留兼容入口。
2. 对同一 5k stratified sample 汇总 seasonal naive、DLinear 和 PatchTST 的 oracle gap，判断互补性。
3. 接入第三个训练型专家建议优先 TSMixer。
