# Stage 1.3a0：Visual Embedding Cache Smoke

## 1. 实验目的

验证 Stage 1.2 `view_tensor [B,3,64,192]` 可以被 visual embedding cache smoke 消费，并生成与 `physical_window_id` 对齐的 visual embedding cache。

本阶段不训练 visual encoder，不实现 router/gate，不读取 expert error，不运行专家模型。当前 adapter 仅为 deterministic smoke，用于稳定输入/输出协议和 latency 记录，不作为正式视觉先验结论。

范围修正：本次产物应记为 Stage 1.3a0，而不是原先定义的 Stage 1.3a adapter comparison。它没有比较 `per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding` 三条路线。

## 2. 实验计划

1. 将 Stage 1.3a0 计划文档叙述层改为中文。
2. 新增 `tests/test_quitobench_visual_encoder_adapter_smoke.py`。
3. 新增 `tools/quitobench_visual_encoder_adapter_smoke.py`。
4. 读取 Stage 1.2 smoke tensor 和 image index。
5. 输出 `embeddings.parquet`、`embedding_index.csv`、`latency.csv` 和 `manifest.json`。
6. 验证 embedding 行数、维度、主键唯一性和非目标项标记。
7. 在 CPU 上做正式 smoke；在 GPU 繁忙条件下补一个 GPU sanity。
8. 与正在运行的 Stage 1.4g-b expert runner 隔离。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期失败：

```text
ModuleNotFoundError: No module named 'tools.quitobench_visual_encoder_adapter_smoke'
```

实现后测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

结果：

```text
5 passed in 4.10s
```

CPU smoke：

```bash
conda run -n quito python tools/quitobench_visual_encoder_adapter_smoke.py \
  --image-tensor-dir outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1 \
  --embedding-dim 64 \
  --batch-size 128 \
  --device cpu
```

结果：

```text
[input] view_tensor_shape=[288, 3, 64, 192] rows=288
[done] output=/home/user10/TSF/DATAPrepare/outputs/vision_ts_routing/visual_embeddings/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1
[done] embeddings_shape=[288, 64]
[done] encoder_latency_ms_per_window=34.9777
```

GPU sanity：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n quito python tools/quitobench_visual_encoder_adapter_smoke.py \
  --image-tensor-dir outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1 \
  --output-root outputs/vision_ts_routing/visual_embeddings_gpu_sanity \
  --embedding-dim 64 \
  --batch-size 128 \
  --device cuda
```

结果：

```text
[input] view_tensor_shape=[288, 3, 64, 192] rows=288
[done] output=outputs/vision_ts_routing/visual_embeddings_gpu_sanity/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1
[done] embeddings_shape=[288, 64]
[done] encoder_latency_ms_per_window=129.1031
```

GPU 状态检查：

```bash
nvidia-smi
```

观察到 4 张 L40 均被其他用户进程占用，且重命名后复测期间 Python/pytest 一度卡在文件页 I/O 等待。因此本次 CPU/GPU latency 只能说明新命名产物路径可运行，不适合作为干净性能结论。此前旧目录名 smoke 中 CPU latency 曾为约 `0.3606 ms/window`，GPU 繁忙条件下 sanity latency 曾为约 `37.6868 ms/window`。

## 4. 输入数据与配置

输入 Stage 1.2 image tensor：

```text
outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1/
```

输入规模：

```text
view_tensor_shape = [288, 3, 64, 192]
sample_set_id = qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
image_protocol_id = view3_h64_w192_v1
view_tensor_semantics = multi_view_not_rgb
```

Adapter 配置：

```text
encoder_id = tiny_view_cnn_v1
embedding_dim = 64
batch_size = 128
random_seed = 20260607
```

CPU 输出：

```text
outputs/vision_ts_routing/visual_embeddings/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1/
```

GPU sanity 输出：

```text
outputs/vision_ts_routing/visual_embeddings_gpu_sanity/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1/
```

## 5. 实验结果

新增文件：

```text
docs/superpowers/plans/2026-06-07-stage1-3a0-visual-embedding-cache-smoke.md
tests/test_quitobench_visual_encoder_adapter_smoke.py
tools/quitobench_visual_encoder_adapter_smoke.py
experiment_logs/2026-06-07_2255_stage1_3a0_visual_embedding_cache_smoke.md
```

CPU 输出文件：

```text
embeddings.parquet
embedding_index.csv
latency.csv
manifest.json
```

CPU latency：

```text
encoder_latency_ms_per_window = 34.9777
```

GPU sanity latency：

```text
encoder_latency_ms_per_window = 129.1031
```

重命名后 CPU/GPU latency 均明显慢于早先 smoke，原因是本次复测期间系统 I/O 和 GPU 均处于重负载状态，且本 smoke 样本数很小，启动和排队开销占比高。

Manifest 关键字段：

```text
trains_visual_encoder = false
runs_expert_models = false
implements_router = false
reads_expert_errors = false
uses_future_target = false
view_tensor_semantics = multi_view_not_rgb
embedding_format = wide_columns
```

## 6. 问题与观察

- 本阶段没有使用 `torchvision` 或 `timm`，避免引入新依赖。
- `TinyViewCnnEncoder` 只是协议 smoke，不代表正式视觉 encoder。
- CPU smoke 结果可以作为当前 Stage 1.3a0 的主要输出。
- 本阶段未覆盖原 Stage 1.3a 定义中的三种 adapter 消费策略：`per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding`。
- GPU sanity 在 GPU 繁忙条件下完成，只能证明 CUDA 路径可运行；干净 GPU latency 需要后续在空闲 GPU 上单独复测。
- 本阶段未修改 `tools/quitobench_framework_expert_cache.py`，也未写入 `outputs/vision_ts_routing/expert_predictions/`。
- Stage 1.4g-b 在本阶段期间仍有 DLinear/PatchTST 任务运行；Stage 1.3a0 输出目录与其隔离。

## 7. 结论

Stage 1.3a0 已形成第一版 visual embedding cache 协议：

```text
Stage 1.2 view_tensor
-> TinyViewCnnEncoder smoke adapter
-> physical_window_id 对齐的 embeddings.parquet
```

该输出可用于后续 Stage 1.5 gate baseline 的接口联调，但不应作为正式视觉结构先验效果结论。

## 8. 下一步计划

1. 在 Stage 1.5 前固定专家池和 oracle target 生成口径。
2. 补做正式 Stage 1.3a adapter comparison：`per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding`。
3. 在 GPU 空闲时复测 encoder forward latency，避免将本次繁忙 GPU sanity 作为性能结论。
