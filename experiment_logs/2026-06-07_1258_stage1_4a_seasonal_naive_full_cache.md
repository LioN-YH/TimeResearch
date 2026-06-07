# Stage 1.4a-expanded：seasonal_naive 全量轻量缓存固化

## 1. 实验目的

在 50k 分层分析确认 `seasonal_naive` 是当前轻量专家池的统治性 baseline 后，对正式 working registry 的 627,430 条 sample-channel window 全量固化 `seasonal_naive` 缓存。

本实验目标是为后续 DLinear/NLinear 等 Stage 1.4b 专家提供稳定、低成本、history-only 的统计 baseline。本窗口不进入 Stage 1.4b，不实现 router，不运行视觉 encoder，不接入神经网络专家。

## 2. 实验计划

1. 在现有轻量专家缓存脚本中增加 `--expert-ids` 参数。
2. 保持默认四专家行为不变；显式传入 `--expert-ids seasonal_naive` 时只生成 `seasonal_naive`。
3. 对完整 `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e` registry 运行全量缓存。
4. 校验输出文件、唯一主键、manifest 边界标志和 split/subset/cell 覆盖。
5. 更新实验日志总览。

## 3. 执行命令

聚焦测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_lightweight_expert_cache.py -q
```

全量 seasonal baseline 缓存：

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --expert-ids seasonal_naive \
  --expert-set-id lightweight_v1__seasonal_naive_full
```

结果分析：

```bash
conda run -n quito python /tmp/stage14a_seasonal_full_analyze.py
```

## 4. 输入数据与配置

- registry: `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv`
- sample_set_id: `qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- expert_set_id: `lightweight_v1__seasonal_naive_full`
- expert_ids: `seasonal_naive`
- windows: `627430`
- 预测输入策略：仅使用 history
- target 使用策略：仅用于 error 和 profiling

## 5. 实验结果

输出目录：

```text
outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full/
```

输出文件：

- `predictions.parquet`
- `errors.parquet`
- `manifest.json`
- `profiling/cell_model_matrix.csv`
- `profiling/oracle_summary.csv`

运行结果：

```text
[done] windows=627430
[done] prediction_rows=627430
[done] latency_ms_per_window=0.5953
```

缓存校验：

```text
manifest_windows 627430
expert_ids ['seasonal_naive']
prediction_rows 627430
error_rows 627430
prediction_unique True
error_unique True
soft_weight_max_abs_error 0.0
implements_router False
runs_visual_encoder False
runs_neural_experts False
latency_ms_per_window 0.5953388241103251
```

split 覆盖：

| split | windows |
| --- | ---: |
| train | 386,220 |
| valid | 96,875 |
| test | 144,335 |

subset 覆盖：

| subset | windows |
| --- | ---: |
| hour | 403,260 |
| min | 224,170 |

cell 覆盖：

| official_tsf_cell | windows | seasonal_naive_mse | seasonal_naive_mae |
| --- | ---: | ---: | ---: |
| highT_highS_highF | 129,480 | 4.692192e+07 | 586.153440 |
| highT_highS_lowF | 102,650 | 3.576051e+10 | 13,937.592280 |
| highT_lowS_highF | 49,300 | 4.696881e+09 | 9,561.655852 |
| highT_lowS_lowF | 47,490 | 2.895989e+09 | 5,873.770575 |
| lowT_highS_highF | 87,760 | 2.840342e+11 | 39,560.602702 |
| lowT_highS_lowF | 113,310 | 1.498731e+11 | 40,069.792784 |
| lowT_lowS_highF | 49,010 | 1.052240e+07 | 281.818331 |
| lowT_lowS_lowF | 48,430 | 1.867935e+10 | 14,570.533143 |

单专家 oracle summary：

| 指标 | 值 |
| --- | ---: |
| num_windows | 627,430 |
| num_experts | 1 |
| oracle_mse | 7.468579e+10 |
| best_fixed_expert | seasonal_naive |
| best_fixed_mse | 7.468579e+10 |
| uniform_ensemble_mse_proxy | 7.468579e+10 |
| oracle_gap_vs_best_fixed | 0.0 |

## 6. 问题与观察

1. 单专家缓存下 oracle、best fixed 和 uniform proxy 三者相同，这是预期行为，不表示存在 ensemble 上界。
2. 全量 MSE 与 50k 分层 MSE 不同，原因是 50k 是 `split/subset/official_tsf_cell` 尽量均衡抽样，而全量 registry 的 train/hour/cell 分布更接近正式 working registry 本身。
3. 当前脚本全量运行的主要成本来自逐窗口抽取 history/target，而不是 `seasonal_naive` 计算。全量单专家耗时折算约 `0.5953 ms/window`，可接受。
4. `lowT_highS_highF` 和 `lowT_highS_lowF` 的全量 MSE 明显更高，后续 DLinear/NLinear 若能在这些 cell 上改善，将是判断正式专家互补性的重点。

## 7. 结论

`lightweight_v1__seasonal_naive_full` 已可作为 Stage 1.4a 的正式 lightweight baseline 缓存。它适合保留到正式专家池中，作为后续 DLinear/NLinear 和 router/gate 分析的统计锚点。

本实验没有实现 router，没有运行视觉 encoder，没有运行神经网络专家，没有进入 Stage 1.4b。

## 8. 下一步计划

1. 后续 Stage 1.4b 的 DLinear/NLinear 输出应复用当前 `predictions/errors/manifest/profiling` 协议。
2. 比较 DLinear/NLinear 时优先报告相对 `seasonal_naive` 的全局、split 和 cell-level 改善。
3. 若需要训练 gate，再把 `lightweight_v1__seasonal_naive_full` 与 Stage 1.4b 专家缓存按 `physical_window_id` 对齐拼接。
