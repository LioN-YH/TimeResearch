# Stage 1.4g-b：官方任务网格稀疏 registry sanity 尝试

## 1. 实验目的

在 Stage 1.4g 发现 Quito CLI / Quito-native exhaustive full-window 路径过慢后，本阶段尝试改用项目内已验证的 registry + expert cache runner，完成两个官方任务网格的稀疏 sanity：

```text
96_48_S
576_288_S
```

目标是并行训练/测试 `DLinear` 与 `PatchTST`，检查它们在更接近 Quito 官方 seq/pred 网格的设置下是否相对 `seasonal_naive` 恢复。

本阶段不实现 router/gate，不运行视觉 encoder，不生成 OOF cache，不修改 Quito 上游源码。

## 2. 实验计划

1. 为 `96_48_S` 生成独立 `quito_overlap` registry；
2. 为 `576_288_S` 生成独立 `quito_overlap` registry；
3. 使用 `sample_stride=288` 保留全 item/channel 的稀疏滑窗；
4. 通过 `tools/quitobench_framework_expert_cache.py` 并行运行：
   - `DLinear 96_48_S`
   - `PatchTST 96_48_S`
   - `DLinear 576_288_S`
   - `PatchTST 576_288_S`
5. 每个任务使用不同 GPU；
6. 第一轮计划使用 `stratified_rows=50000`、`epochs=5`、`train_set_standardize`、cosine scheduler。

## 3. 执行命令

生成 `96_48_S` registry：

```bash
conda run -n quito python tools/quitobench_window_registry.py \
  --history-len 96 \
  --pred-len 48 \
  --sample-stride 288
```

生成 `576_288_S` registry：

```bash
conda run -n quito python tools/quitobench_window_registry.py \
  --history-len 576 \
  --pred-len 288 \
  --sample-stride 288
```

并行训练尝试示例：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506 \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14g_b_h96_p48_stride288_50k_e5 \
  --stratified-rows 50000 \
  --epochs 5 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --learning-rate 1e-4 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 1e-5 \
  --num-workers 4 \
  --device cuda
```

降级 smoke 尝试：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506 \
  --expert-model dlinear \
  --expert-set-id dlinear_v1__stage14g_b_h96_p48_stride288_1k_e1_raw_smoke \
  --max-rows 1000 \
  --epochs 1 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --learning-rate 1e-4 \
  --drop-last \
  --scheduler cosine \
  --eta-min 1e-5 \
  --num-workers 0 \
  --device cuda
```

## 4. 输入数据与配置

数据源：

```text
data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/
```

新 registry：

```text
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/
outputs/vision_ts_routing/window_registry/qb_h576_p288_quito_overlap_d8cfe7ee_stride288_d9655deb/
```

## 5. 实验结果

`96_48_S` registry 已生成：

```text
sample_set_id=qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506
windows=220755
split_window_counts={'train': 133040, 'valid': 36165, 'test': 51550}
subset_window_counts={'hour': 139590, 'min': 81165}
```

`576_288_S` registry 已生成：

```text
sample_set_id=qb_h576_p288_quito_overlap_d8cfe7ee_stride288_d9655deb
windows=194955
split_window_counts={'train': 120140, 'valid': 29715, 'test': 45100}
subset_window_counts={'hour': 129250, 'min': 65705}
```

并行训练尝试：

- 4 个任务均成功启动；
- 每个任务分配不同 `CUDA_VISIBLE_DEVICES`；
- 进程均停在 I/O wait (`D` state)，GPU memory/use 均未上升；
- 等待约 3 分钟后仍未进入训练；
- 判断瓶颈发生在训练前的数据准备阶段，尤其是 registry/parquet/history-target extraction 或 Quito scaler 构造；
- 手动终止，未产出 predictions/errors。

降级 `1k/e1/raw` DLinear smoke：

- 去掉 `--train-set-standardize`；
- 使用 `--max-rows 1000`；
- 使用 `--num-workers 0`；
- 仍停在 I/O wait，超过 3 分钟未进入训练；
- 手动终止，未产出 predictions/errors。

工具增强：

```text
tools/quitobench_quito_native_utils.py
tools/quitobench_quito_native_sanity.py
tests/test_quitobench_quito_native_sanity.py
```

轻量测试结果：

```bash
conda run -n quito python -m pytest tests/test_quitobench_quito_native_sanity.py -q
```

```text
4 passed
```

## 6. 问题与观察

本阶段证明：

1. 稀疏 registry 可以快速生成，说明窗口枚举本身不是阻塞点；
2. 模型训练没有真正进入 GPU 阶段，四张卡空闲；
3. 即使降到 1k raw smoke，`framework_expert_cache` 仍可能在数据抽取阶段进入长 I/O wait；
4. 直接并行启动多任务会放大数据准备阶段的 I/O 压力，不适合当前实现。

当前没有产生 PatchTST/DLinear 在 `96_48_S` 或 `576_288_S` 上的有效指标，因此不能回答它们是否在官方任务网格下恢复。

## 7. 结论

Stage 1.4g-b 未完成训练测试指标产出，但完成了两个官方任务网格的稀疏 registry 固化。

阻塞点已经从“GPU 是否足够”明确转移到“数据准备/历史窗口抽取路径的 I/O wait”。服务器四张 GPU 空闲不能直接解决该问题。

## 8. 下一步计划

建议下一步不要继续直接启动训练，而是先修数据准备路径：

1. 为 `framework_expert_cache` 增加阶段性进度输出，至少区分：
   - 读取 registry；
   - 分层抽样；
   - 读取 parquet；
   - 抽取 history/target；
   - Quito scaler 构造；
   - 训练；
   - 预测；
2. 将已抽取的 `(physical_window_id, history, target, scaler)` 固化为中间 parquet/npz cache；
3. 先单任务生成 `96_48_S` 的 1k/10k data cache；
4. data cache 生成成功后，再并行训练 DLinear/PatchTST；
5. 只有数据准备阶段可复用后，才值得把四张 GPU 全部用于训练。
