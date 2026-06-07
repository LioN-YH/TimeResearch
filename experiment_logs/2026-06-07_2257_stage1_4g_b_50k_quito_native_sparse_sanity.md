# Stage 1.4g-b：Quito-native sparse registry 50k sanity

## 1. 目的

继续验证官方任务网格：

```text
96_48_S
576_288_S
```

使用项目内 registry + Quito 模型 wrapper，对 `seasonal_naive`、`DLinear`、`PatchTST` 做同口径 50k 分层窗口 sanity。

本轮不是官方 full-window exhaustive 复现；它仍是固定 stride registry 上的交互式 sanity，用来判断之前 PatchTST/DLinear 弱于 seasonal_naive 是否只是小样本或窗口长度错配导致。

## 2. 代码修复与诊断

本轮先修复/增强：

1. `tools/quitobench_framework_expert_cache.py`
   - 增加 `--progress-every` 阶段日志；
   - 自动从 registry 推断 `seq_len/pred_len`，避免 `96_48_S` 仍使用默认 `192/96`；
   - 默认输出目录按当前 `sample_set_id` 派生，避免落到旧的 `h192_p96` 目录。
2. `tools/quitobench_lightweight_expert_cache.py`
   - `extract_histories_and_targets()` 增加进度日志；
   - parquet 读取后一次性构造 subset/item lookup，避免每个新 item 重复全表 boolean filter。

关键定位：

```text
100-row smoke 原始失败：
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x96 and 192x96)
```

根因是模型配置窗口长度未跟随 registry。修复后 100-row DLinear smoke 成功完成。

## 3. 输入与命令口径

Registry：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/
outputs/vision_ts_routing/window_registry/qb_h576_p288_quito_overlap_d8cfe7ee_stride288_d9655deb/
```

训练口径：

```text
stratified_rows=50000
stratify_cols=split,subset,official_tsf_cell
epochs=5
scheduler=cosine
eta_min=1e-5
num_workers=0
drop_last=True
```

多卡执行说明：

- 起初四路 `conda run` 并行时 stdout 被缓冲，且一度进入 XFS/page-cache wait；
- 后改为 direct Python + `PYTHONUNBUFFERED=1`；
- GPU 实际被其他用户任务占用较多，但本轮模型显存占用较小，仍完成四路/多路执行。

后台日志：

```text
outputs/vision_ts_routing/run_logs/dlinear_h576_p288_50k_e5.log
outputs/vision_ts_routing/run_logs/patchtst_h96_p48_50k_e5.log
outputs/vision_ts_routing/run_logs/patchtst_h576_p288_50k_e5.log
```

汇总 CSV：

```text
outputs/vision_ts_routing/stage14g_b_50k_summary.csv
```

## 4. 50k 结果

| grid | model | windows | train_windows | MSE | MAE | MSE/seasonal | MAE/seasonal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 96_48_S | seasonal_naive | 50000 | - | 1.401926e+11 | 3.263713e+04 | 1.000 | 1.000 |
| 96_48_S | DLinear | 50000 | 20750 | 5.037314e+11 | 5.224542e+04 | 3.593 | 1.601 |
| 96_48_S | PatchTST | 50000 | 20750 | 3.920135e+13 | 5.491413e+05 | 279.625 | 16.826 |
| 576_288_S | seasonal_naive | 50000 | - | 4.572875e+10 | 1.537339e+04 | 1.000 | 1.000 |
| 576_288_S | DLinear | 50000 | 23996 | 1.349768e+11 | 2.926184e+04 | 2.952 | 1.903 |
| 576_288_S | PatchTST | 50000 | 23996 | 4.646468e+14 | 1.318092e+06 | 10160.933 | 85.739 |

输出目录：

```text
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/
outputs/vision_ts_routing/expert_predictions/qb_h576_p288_quito_overlap_d8cfe7ee_stride288_d9655deb/
```

## 5. Quito train-set standardize smoke

为了检查 raw-scale 是否导致神经模型异常，补跑了 `96_48_S` 的 1k/e1 `--train-set-standardize` smoke。

结果：

| model | MSE | MAE | elapsed_s | train_windows | standardization |
|---|---:|---:|---:|---:|---|
| seasonal_1k | 2.635543e+10 | 17287.020298 | 9.660130 | - | - |
| DLinear raw 1k | 4.662695e+11 | 74643.899372 | 12.885652 | 336 | False |
| PatchTST raw 1k | 3.560679e+11 | 65375.620461 | 15.569276 | 336 | False |
| DLinear Quito-std 1k | 4.952874e+11 | 74297.876866 | 136.176669 | 336 | True |
| PatchTST Quito-std 1k | 3.655945e+11 | 65241.071832 | 138.106347 | 336 | True |

观察：

- Quito `TimeSeriesDataset` 构造两个 subset 的 scaler 约需 130 秒；
- 1k/e1 下标准化没有改善数量级；
- 因此本轮反常不能简单归因于 raw-scale wrapper。

## 6. 结论

1. 之前的 1k raw smoke 有一个明确 bug：模型窗口长度没有跟随官方任务网格，已修复。
2. 修复后，50k/e5 下 `DLinear/PatchTST` 仍弱于 `seasonal_naive`。
3. `PatchTST` 在当前 wrapper/training budget 下明显发散或严重欠拟合，不适合作为恢复结论。
4. 这轮结果支持一个更具体的判断：QuitoBench 这些窗口的强周期性使 `seasonal_naive` 非常强；当前“单一全局神经模型 + 5 epoch + sparse stratified windows”的设置不足以超过它。
5. 这仍不是官方 finetune/tune/evaluate full-window 复现；下一步若要继续追官方口径，应优先复用/缓存 Quito scaler 与抽取结果，再做更长训练或官方 trainer 复现。

## 7. 验证

```text
py_compile tools/quitobench_framework_expert_cache.py tools/quitobench_lightweight_expert_cache.py: passed
100-row DLinear smoke: passed
1k DLinear/PatchTST x 2 grids: passed
50k DLinear/PatchTST x 2 grids: passed
50k seasonal_naive x 2 grids: passed
```
