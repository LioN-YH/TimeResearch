# Stage 1.4g：Quito 原生官方任务网格 sanity 尝试

## 1. 实验目的

用 Quito 原生数据与模型封装校准 Stage 1.4f 中 `PatchTST`、`DLinear` 弱于 `seasonal_naive` 的现象，优先检查两个官方任务配置：

```text
96_48_S
576_288_S
```

本阶段不实现 router/gate，不运行视觉 encoder，不生成 OOF cache，不修改 Quito 上游源码。

## 2. 实验计划

原计划：

1. 使用 Quito CLI 跑 `snaive` 的 `evaluate`；
2. 使用 Quito CLI 多 GPU 跑 `DLinear/PatchTST` 的 `finetune`；
3. 对训练型模型 checkpoint 跑 `evaluate`；
4. 汇总 `MSE/MAE/MASE/SMAPE`，判断 Quito 原生流程下 PatchTST/DLinear 是否恢复。

用户补充机器有多 GPU，因此本次优先尝试：

```text
evaluate: --num_processes 4 --use_gpu 1
finetune: --num_processes 4 --use_gpu 1
```

## 3. 执行命令

数据路径对齐：

```bash
mkdir -p quito/examples/datasets/cluster_data
ln -sfn ../../../../data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet \
  quito/examples/datasets/cluster_data/open_hour_data.parquet
ln -sfn ../../../../data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet \
  quito/examples/datasets/cluster_data/open_min_data.parquet
```

Quito CLI 单 GPU evaluate 尝试：

```bash
cd quito
conda run -n quito quito-cli evaluate \
  --config_path configs/evaluate/snaive/96_48_S.yaml \
  --num_processes 1 \
  --use_gpu 1
```

Quito CLI 多 GPU evaluate 尝试：

```bash
cd quito
conda run -n quito quito-cli evaluate \
  --config_path configs/evaluate/snaive/96_48_S.yaml \
  --num_processes 4 \
  --use_gpu 1
```

为绕开 Ray per-user deepcopy 开销，新增 Quito-native batch sanity runner：

```bash
conda run -n quito python tools/quitobench_quito_native_sanity.py \
  --config-path quito/configs/evaluate/snaive/96_48_S.yaml \
  --output-dir outputs/vision_ts_routing/quito_native_sanity/snaive__96_48_S__smoke \
  --device cuda:0 \
  --max-batches 2
```

全量 `snaive 576_288_S` batch runner 尝试：

```bash
conda run -n quito python tools/quitobench_quito_native_sanity.py \
  --config-path quito/configs/evaluate/snaive/576_288_S.yaml \
  --output-dir outputs/vision_ts_routing/quito_native_sanity/snaive__576_288_S \
  --device cuda:0 \
  --eval-batch-size 8192
```

## 4. 输入数据与配置

Quito 原生配置：

```text
quito/configs/evaluate/snaive/96_48_S.yaml
quito/configs/evaluate/snaive/576_288_S.yaml
quito/configs/finetune/dlinear/96_48_S.yaml
quito/configs/finetune/dlinear/576_288_S.yaml
quito/configs/finetune/patchtst/96_48_S.yaml
quito/configs/finetune/patchtst/576_288_S.yaml
```

本地数据：

```text
data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet
data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet
```

GPU 环境：

```text
torch 2.6.0+cu124
CUDA available = True
GPU count = 4
GPU = NVIDIA L40 x4
```

## 5. 实验结果

### 5.1 Quito CLI evaluate 结果

`snaive 96_48_S` 单 GPU / 单 actor：

- 输出目录：`quito/outputs/snaive/96_48_S/EVALUATE/ver_0/`
- 日志显示已加载 `TEST_DATA_MIN` 和 `TEST_DATA_HOUR`；
- Ray 创建 `1` 个 evaluator；
- `Total tasks to evaluate: 1290`；
- 运行约 3 分钟无进度输出，确认一个 actor 在执行 `ModelEvaluator.evaluate_user`；
- 手动终止。

`snaive 96_48_S` 四 GPU / 四 actor：

- 输出目录：`quito/outputs/snaive/96_48_S/EVALUATE/ver_1/`
- 日志显示 Ray 创建 `4` 个 evaluator；
- Ray 资源识别为 `GPU,4`；
- `Total tasks to evaluate: 1290`；
- 运行约 2 分钟后仍停留在前 4 个 `ModelEvaluator.evaluate_user`；
- GPU 利用率接近 0；
- 主要瓶颈不是 GPU，而是 Quito `evaluate.py` 对每个 user 做 `deepcopy(dataset)` 的 per-user 调度方式；
- 手动终止。

### 5.2 Quito-native batch runner 结果

新增脚本：

```text
tools/quitobench_quito_native_sanity.py
tests/test_quitobench_quito_native_sanity.py
```

测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_quito_native_sanity.py -q
```

结果：

```text
2 passed
```

`snaive 96_48_S` 2 batch smoke：

```text
output_dir=outputs/vision_ts_routing/quito_native_sanity/snaive__96_48_S__smoke
model=NaiveForecaster
samples=256
elapsed_seconds=49.36
MSE=2.8623778820
MAE=1.1496376991
MASE=1.8997653127
MASE_LEAK=2.2834629416
MAPE=467.6637878418
SMAPE=137.9857406616
SMASE=1.0071137547
```

`snaive 576_288_S` 全量 batch runner：

- 使用 `eval_batch_size=8192`；
- 主进程 CPU 持续增长，内存约 `5.4GB`；
- 运行超过 12 分钟仍未完成；
- 判断为千万级窗口 exhaustive evaluation 的 DataLoader 吞吐瓶颈；
- 手动终止，未产出最终全量指标。

## 6. 问题与观察

本次关键发现是：所谓“QuitoBench 全量训练测试”不是轻量 sanity。

按 Quito `TimeSeriesDataset` 的 sample-window 口径估算，`S` 模式每个 item/channel 都参与滑窗：

- `96_48_S` test 约千万级窗口；
- `576_288_S` test 也约千万级窗口；
- train split 每个 epoch 约三千万级窗口；
- 官方 finetune YAML 默认 `num_epochs=5`、`batch_size=128`，即使用 4 GPU，完整训练也会是长任务。

Quito CLI evaluate 的瓶颈不是 GPU，而是：

```text
每个 user 任务 deepcopy(dataset) -> 单 user DataLoader -> Ray actor
```

因此提高 `--num_processes` 能并行 actor，但不能消除每个 actor 上的大对象拷贝和逐 user 调度开销。

## 7. 结论

本阶段未完成 `96_48_S` / `576_288_S` 的完整官方全量训练测试。

已完成：

- 对齐 Quito 原生配置的数据路径；
- 确认 Quito CLI 多 GPU evaluate 可启动并识别 4 张 L40；
- 确认官方 evaluate 流程在本地对全量窗口 sanity 过慢；
- 新增并测试 Quito-native batch sanity runner；
- 跑通 `snaive 96_48_S` 的 2-batch smoke。

当前不能据此判断 PatchTST/DLinear 在官方任务网格下是否恢复。

## 8. 下一步计划

建议不要直接用官方 YAML 做完整 `5 epoch x full windows` 交互式实验。更稳的下一步是单独规划 Stage 1.4g-b：

1. 为 Quito-native runner 增加进度输出和 `stride / max_windows_per_item` 参数；
2. 先跑 `96_48_S` 与 `576_288_S` 的固定 stride 全 item sanity；
3. 对 DLinear/PatchTST 使用相同窗口集合训练和评估；
4. 若固定 stride 结果显示 PatchTST/DLinear 恢复，再决定是否启动 overnight/full exhaustive 训练；
5. 若必须 full exhaustive，应作为长任务运行，并明确记录预计步数、GPU 数、checkpoint 输出和中断恢复策略。
