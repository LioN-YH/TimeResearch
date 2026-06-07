# Stage 1.4b：DLinear 正式训练型专家接入 smoke

## 1. 目的

验证第一个正式训练型专家能否接入 Stage 1.4 专家预测缓存协议。专家训练只使用 `train` split；预测缓存覆盖 `train/valid/test`，供后续 router/gate 训练、验证和最终测试分别使用。

本阶段只做 DLinear smoke，不实现 router，不运行视觉 encoder，不修改 Quito 上游代码，不执行 Stage 1.4a-expanded 分析。

## 2. 框架审计结论

- Quito 主仓库路径：`/home/user10/TSF/DATAPrepare/quito`
- Quito 已包含 `quito.models.dlinear.DLinear`
- Quito 已包含 `quito.datasets.TimeSeriesDataset` 和 `quito-cli finetune/evaluate` runner
- 本地存在 `Time-Series-Library`：`/home/user10/TSF/Time-Series-Library`
- 当前 `quito` conda 环境中 `tslib` 包不可导入
- 第一版 smoke 选择 Quito DLinear 模型本体 + 本项目 registry thin wrapper

选择该路径的原因：

1. Quito DLinear 与当前 QuitoBench split 口径来源一致。
2. 直接复用 Stage 1.0 registry 的 `physical_window_id`，不用先改造上游 runner 输出样本编号。
3. 能在短 smoke 中明确验证 train/valid/test、prediction/error schema 和 manifest。

## 3. 输入

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- expert_set_id: `dlinear_v1__smoke`
- max_rows: `2000`
- max_train_windows: `256`
- epochs: `1`
- batch_size: `64`
- device: `cpu`

## 4. 实现范围

新增：

- `tools/quitobench_dlinear_expert_cache.py`
- `tests/test_quitobench_dlinear_expert_cache.py`

复用 Stage 1.4a：

- `extract_histories_and_targets()`
- `compute_error_table()`
- `compute_oracle_summary()`
- `build_cell_model_matrix()`
- `write_expert_cache_outputs()`

## 5. 命令

单元测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py -q
```

smoke 缓存：

```bash
conda run -n quito python tools/quitobench_dlinear_expert_cache.py \
  --max-rows 2000 \
  --max-train-windows 256 \
  --epochs 1 \
  --batch-size 64 \
  --expert-set-id dlinear_v1__smoke \
  --device cpu
```

smoke 校验：

```bash
conda run -n quito python -c "import json; from pathlib import Path; import pandas as pd; out=Path('outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__smoke'); manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')); pred=pd.read_parquet(out/'predictions.parquet'); err=pd.read_parquet(out/'errors.parquet'); print('windows', manifest['total_windows']); print('prediction_rows', len(pred)); print('error_rows', len(err)); print('prediction_unique', pred[['physical_window_id','expert_id']].duplicated().sum() == 0); print('error_unique', err[['physical_window_id','expert_id']].duplicated().sum() == 0); print('splits', pred['split'].value_counts().to_dict()); print('train_stats', manifest['training_stats']); print('soft_weight_max_abs_error', float((err.groupby('physical_window_id')['soft_oracle_weight'].sum() - 1.0).abs().max())); print('implements_router', manifest['implements_router']); print('runs_visual_encoder', manifest['runs_visual_encoder']); print('runs_neural_experts', manifest['runs_neural_experts']); print('source_framework', manifest['source_framework']); print('audit', manifest['audit_summary'])"
```

## 6. 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__smoke/
```

包含：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

## 7. 验证结果

新增单元测试：

```text
5 passed
```

Stage 1.4a + Stage 1.4b 相关测试：

```text
10 passed
```

smoke 输出：

```text
[done] output_dir=/home/user10/TSF/DATAPrepare/.worktrees/stage1-4b-framework/outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__smoke
[done] windows=631
[done] prediction_rows=631
[done] train_windows=256
```

smoke 校验：

```text
windows 631
prediction_rows 631
error_rows 631
prediction_unique True
error_unique True
splits {'valid': 300, 'train': 256, 'test': 75}
train_stats {'train_windows': 256, 'trained_splits': ['train'], 'epochs_completed': 1, 'final_train_loss': 0.4742293059825897, 'train_elapsed_seconds': 0.2960586315020919, 'device': 'cpu'}
soft_weight_max_abs_error 0.0
implements_router False
runs_visual_encoder False
runs_neural_experts True
source_framework quito
```

## 8. 结论

Stage 1.4b DLinear smoke 跑通。

当前已经验证：

- Quito DLinear 可以作为第一个正式训练型专家接入；
- 训练只使用 train split；
- `train/valid/test` 均生成预测缓存和误差缓存；
- `valid/test` 不参与专家训练，只参与推理、误差和 oracle 计算；
- 预测能映射回 `physical_window_id`；
- 输出复用 Stage 1.4a 的 `predictions/errors/manifest/profiling` schema；
- 未实现 router；
- 未运行视觉 encoder。

## 9. 下一步

1. 可把 DLinear smoke 扩展为分层抽样版本，增加 train/valid/test 中不同 subset 和 official TSF cell 覆盖。
2. 可接入 NLinear 或 PatchTST，继续复用同一缓存协议。
3. 若后续需要直接使用 Quito runner，应先实现上游样本顺序到 `physical_window_id` 的显式映射校验。
