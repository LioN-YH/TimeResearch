# Stage 1.4a：轻量专家预测缓存 smoke

## 1. 目的

验证 Stage 1.4a 专家预测缓存 schema、history-only 轻量专家、误差计算、soft oracle 和 cell-level profiling。

本阶段明确不实现 router，不运行视觉 encoder，不运行神经网络专家，不修改 Quito 上游代码。

## 2. 输入

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- max_rows: `512`
- expert_set_id: `lightweight_v1__smoke_max_rows_512`

## 3. 实现范围

新增：

- `tools/quitobench_lightweight_expert_cache.py`
- `tests/test_quitobench_lightweight_expert_cache.py`

轻量专家：

- `last_value`
- `seasonal_naive`
- `recent_mean`
- `linear_trend`

预测函数只接收 history。target 只在 `compute_error_table()` 中用于误差和 oracle 计算。

## 4. 命令

单元测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

smoke 缓存：

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --max-rows 512 \
  --expert-set-id lightweight_v1__smoke_max_rows_512
```

smoke 校验：

```bash
conda run -n quito python -c "import json; from pathlib import Path; import pandas as pd; out=Path('outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512'); manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')); pred=pd.read_parquet(out/'predictions.parquet'); err=pd.read_parquet(out/'errors.parquet'); print('windows', manifest['total_windows']); print('prediction_rows', len(pred)); print('error_rows', len(err)); print('prediction_unique', pred[['physical_window_id','expert_id']].duplicated().sum() == 0); print('error_unique', err[['physical_window_id','expert_id']].duplicated().sum() == 0); print('soft_weight_max_abs_error', float((err.groupby('physical_window_id')['soft_oracle_weight'].sum() - 1.0).abs().max())); print('implements_router', manifest['implements_router']); print('runs_neural_experts', manifest['runs_neural_experts'])"
```

## 5. 输出

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512/
```

包含：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

## 6. 验证结果

单元测试：

```text
5 passed
```

smoke 输出：

```text
[done] output_dir=/home/user10/TSF/DATAPrepare/outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__smoke_max_rows_512
[done] windows=512
[done] prediction_rows=2048
[done] latency_ms_per_window=11.9087
```

smoke 校验：

```text
windows 512
prediction_rows 2048
error_rows 2048
prediction_unique True
error_unique True
soft_weight_max_abs_error 3.3306690738754696e-16
implements_router False
runs_neural_experts False
```

## 7. 执行中修正

1. 原计划中 `recent_mean_fraction=0.25` 与 toy 期望值冲突。当前实现保持默认取 history 末尾 25%，toy 序列 `[1,2,3,4,10,20,30,40]` 对应最近 2 个点均值 `35`。
2. 原计划中 `linear_trend` 的 toy 手算期望不准确。当前测试按实际最小二乘公式锁定预测值 `[38.92857143, 44.52380952, 50.11904762, 55.71428571]`。
3. `soft_oracle_weight` 初版使用 `groupby.apply`，在当前 pandas 版本对单组返回宽表，已改为按 group index 显式赋值。
4. `load_subset_frames()` 需要显式传入 subset 列表，已改为从 registry 的 `subset` 列推断。
5. QuitoBench 原始帧排序列为 `date_time`，不是 `date`；当前实现已对齐 Stage 1.1/1.2 的 `date_time` 排序和 item/channel 缓存方式。

## 8. 结论

Stage 1.4a smoke 通过。当前已经有与 `physical_window_id` 对齐的轻量专家 prediction/error/oracle 缓存路径，可用于后续：

- 扩大到正式 working registry；
- 或先接入 Stage 1.3a visual encoder adapter smoke；
- 或进入 Stage 1.4b 复用 Quito / Time-Series-Library/tslib 接入神经网络专家。

当前仍未实现 router，未运行视觉 encoder，未运行神经网络专家。

