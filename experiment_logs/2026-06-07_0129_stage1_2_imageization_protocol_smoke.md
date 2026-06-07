# Stage 1.2：伪图像协议与视觉输入 Smoke

## 1. 实验目的

实现并验证 Stage 1.2 的 sample-channel view tensor 协议：

- 输入为 Stage 1.0 正式 working registry 中的 sample-channel history window。
- 只读取 history `[history_start_idx, history_end_idx)`，不读取 future target。
- 输出 `view_tensor [N, V, H, W]`，第一版 `V=3,H=64,W=192`。
- 三个 view 分别为 `line_raster / period_fold / fft_power`。
- `V` 是 view dimension，不是自然图像 RGB channel。
- 保持通道独立，继续以 `physical_window_id` 为主键。
- 验证输出可以按 `physical_window_id` join Stage 1.1 proxy。

本阶段不训练视觉 encoder，不抽取 embedding，不运行专家模型，不实现 router。

## 2. 实验计划

1. 在 `docs/superpowers/plans/` 写 Stage 1.2 实现计划。
2. 按 TDD 新增 `tests/test_quitobench_imageization_protocol.py`。
3. 新增 `tools/quitobench_imageization_protocol.py`。
4. 实现 per-window history-only instance normalization。
5. 实现 torch tensor 路径：
   - `line_raster`
   - `period_fold`
   - `fft_power`
6. 从正式 registry 按 `subset/split/official_tsf_cell` 分层抽样 smoke 窗口。
7. 输出 smoke tensor、index、manifest 和少量 debug PNG。
8. 读回输出做完整性验证。
9. 写实验日志并更新总览。

## 3. 执行命令

依赖检查：

```bash
conda run -n quito python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
conda run -n quito python -c "from PIL import Image; print('PIL ok')"
```

结果：

```text
torch 2.12.0+cu130
torch.cuda.is_available() = False
PIL ok
```

CUDA 不可用的原因：

```text
CUDA initialization: The NVIDIA driver on your system is too old (found version 12040)...
```

因此本次 Stage 1.2 只记录 CPU smoke latency，未测 GPU latency。

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

预期失败：

```text
ModuleNotFoundError: No module named 'tools.quitobench_imageization_protocol'
```

实现后 Stage 1.2 测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

结果：

```text
4 passed in 3.50s
```

全量测试：

```bash
conda run -n quito python -m pytest tests -q
```

结果：

```text
36 passed in 2.85s
```

Stage 1.2 smoke：

```bash
conda run -n quito python tools/quitobench_imageization_protocol.py \
  --max-per-group 8 \
  --debug-png-count 16 \
  --device cpu
```

结果：

```text
[input] registry_rows=627430 sampled_rows=288 subsets=('hour', 'min')
[done] output=outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1
[done] tensor_shape=[288, 3, 64, 192]
[done] latency_ms_per_window=0.3107
[done] proxy_join_rows=288
```

读回验证：

```bash
conda run -n quito python -c "<读取 manifest/image_index/view_tensor_sample.npz 并断言 shape、主键、join 和数值范围>"
```

结果：

```text
tensor_shape (288, 3, 64, 192)
index_rows 288
unique_physical True
sample_set_id 1 qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e
finite True
range 0.0 1.0
semantics multi_view_not_rgb
norm_scope per_physical_window_id_history
proxy_join_rows 288
png_count 16
```

## 4. 输入数据与配置

输入 registry：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
```

输入 proxy：

```text
outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/sample_channel_proxy.parquet
```

输入原始数据：

```text
data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet
data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet
```

配置：

```text
image_protocol_id = view3_h64_w192_v1
height = 64
width = 192
view_names = line_raster, period_fold, fft_power
norm_method = instance_mean_std
normalization_scope = per_physical_window_id_history
norm_const = 0.4
eps = 1e-5
clip_min = -5.0
clip_max = 5.0
max_per_group = 8
random_seed = 20260607
debug_png_count = 16
device = cpu
```

## 5. 实验结果

新增文件：

```text
tools/quitobench_imageization_protocol.py
tests/test_quitobench_imageization_protocol.py
docs/superpowers/plans/2026-06-07-stage1-2-imageization-protocol.md
```

输出目录：

```text
outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1/
```

输出文件：

```text
view_tensor_sample.npz   5.6M
image_index.csv          67K
manifest.json            7.6K
debug_png/               16 files
```

输出规模：

```text
sampled_windows = 288
view_tensor shape = [288, 3, 64, 192]
```

关键校验：

- `physical_window_id` 唯一。
- `sample_set_id` 保留且唯一。
- tensor 数值全为有限值。
- tensor 数值范围为 `[0.0, 1.0]`。
- `view_tensor_semantics = multi_view_not_rgb`。
- `normalization.scope = per_physical_window_id_history`。
- proxy join 为 288/288。
- debug PNG 为从 tensor 后处理得到的三联灰度图，不是正式 imageization 路径。

## 6. 问题与观察

- 本次环境中 torch 可用，但 CUDA 不可用，原因是 NVIDIA driver 版本低于当前 PyTorch CUDA build 要求。因此本阶段只记录 CPU latency；GPU latency 需要后续在驱动/torch 匹配的环境中补测。
- `max_per_group=8` 理论上最多 `2 subset * 3 split * 8 cell * 8 = 384`，实际抽到 288。原因是正式 working registry 中并非每个 `subset/split/official_tsf_cell` group 都有样本。
- 三视图 debug PNG 已人工检查一张，非空，能看到左侧 line raster、中间 period fold、右侧 FFT power。
- 当前 `line_raster` 是 soft point raster，尚未连接相邻点形成完整 soft line。该实现足以用于 Stage 1.2 协议 smoke；是否改成连续线段可放到 Stage 1.2b 或 Stage 1.3 前的视图消融中。
- `period_fold` 对 `min` 的 192 点 history 会 padding 到 288，即 2 个 144 周期；padding length 已写入 `image_index.csv`。

## 7. 结论

Stage 1.2 smoke 已完成。

当前已经有可复现的 sample-channel view tensor 协议：

```text
physical_window_id -> history-only normalization -> view_tensor [V=3,H=64,W=192]
```

该协议保留了 `physical_window_id` 和 `sample_set_id`，能与 Stage 1.1 proxy 对齐，并明确记录三视图不是 RGB。后续可以基于该协议进入 Stage 1.3 的视觉 encoder 表征 / 预训练设计。

## 8. 下一步计划

1. 进入 Stage 1.3 前，先决定视觉 encoder 的输入消费策略：
   - per-view grayscale repeat；
   - learned view adapter；
   - custom patch embedding。
2. 后续如需接 frozen ImageNet ViT，必须单独记录 adapter 方案，不能把 `[V,H,W]` 当作 RGB。
3. 在可用 GPU 环境中补测 imageization latency。
4. 继续不要实现 router，不运行专家模型，直到视觉 encoder 和专家预测缓存接口稳定。

## 9. 追加记录：CUDA 环境修复后的 GPU latency

2026-06-07 09:55 重新检查 `quito` 环境，CUDA 已可用：

```bash
nvidia-smi
conda run -n quito python -c "import torch; print('torch', torch.__version__); print('compiled cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
```

结果：

```text
Driver Version: 550.127.08
nvidia-smi CUDA Version: 12.4
GPU: NVIDIA L40 x4
torch 2.6.0+cu124
compiled cuda 12.4
cuda available True
device count 4
device0 NVIDIA L40
```

随后用 GPU 重新运行 Stage 1.2 smoke：

```bash
conda run -n quito python tools/quitobench_imageization_protocol.py \
  --max-per-group 8 \
  --debug-png-count 16 \
  --device cuda
```

结果：

```text
[input] registry_rows=627430 sampled_rows=288 subsets=('hour', 'min')
[done] output=outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1
[done] tensor_shape=[288, 3, 64, 192]
[done] latency_ms_per_window=1.4718
[done] proxy_join_rows=288
```

读回校验：

```text
validated (288, 3, 64, 192) device cuda gpu True ms_per_window 1.4718023497456063
```

补充测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
```

结果：

```text
4 passed in 2.25s
```

观察：

- 当前 GPU latency 是 288 个 smoke 窗口、单次进程、`cuda:0` 下的 measured tensor imageization latency。
- 该数值高于 CPU smoke 的 `0.3107 ms/window`，主要可能来自小 batch 下 GPU kernel launch / synchronization / tensor 拼接开销；它不能代表大 batch 或训练时常驻 GPU pipeline 的吞吐上限。
- Stage 1.3 如果需要严肃比较在线 latency，应单独做 batch size sweep，并区分数据搬运、normalization、三视图构造、encoder forward 的耗时。
