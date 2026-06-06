# Stage 1.2：伪图像协议与视觉输入 Smoke 设计

## 1. 设计目的

Stage 1.2 的目标不是训练视觉 encoder，也不是实现 router，而是在大规模视觉表征实验前，先把 sample-channel history window 到视觉输入张量的协议稳定下来。

本阶段要回答的问题：

> 对同一个 `physical_window_id`，如何只使用 history window，稳定、可复现、低成本地生成视觉输入张量，并验证该张量能与 Stage 1.0 registry、Stage 1.1 proxy 对齐。

本阶段明确不做：

- 不训练 ViT / CNN / MAE。
- 不抽取 visual embedding。
- 不实现 router / gate。
- 不运行专家模型。
- 不生成专家预测缓存。
- 不大规模落盘 PNG。
- 不引入 GAF / RP / CWT 等额外图像化方法作为主线。

## 2. 当前上下文

已完成资产：

- Stage 1.0 working registry：

```text
outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/window_index.csv
```

- Stage 1.1 light proxy：

```text
outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/sample_channel_proxy.parquet
```

关键约束：

- `physical_window_id` 是 proxy、伪图像、专家预测缓存的主键。
- `sample_set_id` 表示当前采样集合，Stage 1.2 必须保留。
- 当前正式 working set 为 `quito_overlap + sample_stride=96`，共 627,430 条 sample-channel 窗口。
- Stage 1.2 只做 smoke，不直接处理全部 627,430 条窗口。
- 当前仍采用通道独立策略：每个输入单位是单个 sample-channel history window。

## 3. 参考原则

Stage 1.2 借鉴以下方向，但不在本阶段复现完整方法：

- ViTime / 视觉范式 TSF：说明 time-series visual representation 方向可行，但本阶段不绑定具体 ViTime 实现。
- TimesNet：按 period 折叠为 2D tensor，用于显式表达周期内和周期间变化。
- VisionTS：采用视觉 masked autoencoder 思路，并在官方实现中使用 instance mean/std normalization、channel independent 处理和周期性 reshape。

参考链接：

- TimesNet: https://arxiv.org/abs/2210.02186
- TimesNet OpenReview: https://openreview.net/pdf?id=ju_Uqw384Oq
- VisionTS: https://arxiv.org/abs/2408.17253
- VisionTS official repo: https://github.com/Keytoyze/VisionTS
- VisionTS model implementation: https://raw.githubusercontent.com/Keytoyze/VisionTS/main/visionts/model.py
- ViTime: https://arxiv.org/abs/2407.07311

## 4. 核心设计决策

### 4.1 继续通道独立

Stage 1.2 输入粒度保持为 sample-channel：

```text
原始 item 矩阵: [L_total, 5]
单个 registry row: (subset, item_id, channel, history_start_idx, history_end_idx)
Stage 1.2 输入: x_history: [history_len]
Stage 1.2 输出: view_tensor: [V, H, W], 第一版 V=3
```

`ind_1` 到 `ind_5` 不在 Stage 1.2 内融合。后续如果需要多通道版本，应作为单独消融或后续阶段，例如：

- multichannel period-fold tensor；
- channel-as-view tensor；
- VisionTS++ 风格 multivariate imageization；
- 通道融合 encoder。

这些内容不进入 Stage 1.2 主实现。

### 4.2 输出是 view tensor，不是 RGB 图片

第一版输出固定为：

```text
view_tensor: [V, H, W]
V = 3
```

这 3 个维度是 view dimension，不是自然图像 RGB channel：

| view index | 名称 | 作用 |
| ---: | --- | --- |
| 0 | `line_raster` | 保留显式时间轴和局部形状 |
| 1 | `period_fold` | 表达周期内 / 周期间结构 |
| 2 | `fft_power` | 表达轻量频域能量分布 |

三个 view 的横纵轴语义并不完全相同：

| view | 横轴含义 | 纵轴含义 |
| --- | --- | --- |
| `line_raster` | time index | normalized value height |
| `period_fold` | phase within period | cycle / period block |
| `fft_power` | frequency bin | repeated or rasterized frequency power |

因此沿 `V` 维堆叠只是统一张量容器，不表示三个 view 在同一个 `(h, w)` 位置有 RGB 式的像素对齐语义。文档和 manifest 中统一使用 `view_tensor`、`view_dim` 或 `view_names`，避免称为 RGB。

后续接视觉 encoder 时必须显式选择 view 消费方式，而不是默认套用 RGB 假设：

- `per_view_grayscale_repeat`：每个 view 单独 repeat 成 `[3, H, W]`，分别过 frozen RGB ViT，再融合 embedding。
- `learned_view_adapter`：用小型 `1x1 conv` 或 patch embedding adapter 将 `[V, H, W]` 映射到 encoder 输入。
- `custom_patch_embedding`：直接让 patch embedding 接收 `V` 个 view，适合后续训练或微调。

debug PNG 不保存为 RGB 混色图作为主解释口径。推荐保存：

- 单 view 灰度图；
- 或三联灰度图；
- 可选另存彩色合成图，但只作快速浏览，不作为正式解释依据。

### 4.3 VisionTS-like per-window history-only instance normalization

Stage 1.2 第一版归一化采用 VisionTS-like mean/std instance normalization。这里的 instance 指单个 `physical_window_id` 对应的 sample-channel history window，不是全数据集统计，也不是同一个 item 的完整历史统计。

```text
对每个 physical_window_id:
x_history = 当前 sample-channel 的 history [history_len]
mean = mean(x_history)
std = std(x_history, unbiased=False)
x_norm = (x_history - mean) / (std / norm_const)
norm_const = 0.4
x_norm = clip(x_norm, clip_min, clip_max)
```

默认配置：

```text
norm_method: instance_mean_std
norm_const: 0.4
eps: 1e-5
clip_min: -5.0
clip_max: 5.0
```

注意：

- 归一化严禁读取 future target。
- batch 实现中，若输入为 `[B, L]`，则 `mean/std` 的 shape 应为 `[B, 1]`，每个窗口独立计算。
- `std < eps` 时使用 `eps`，避免除零。
- `mean/std/norm_const/clip_min/clip_max` 必须写入 metadata。
- Stage 1.1 中的 median/IQR proxy 继续保留为统计特征，但不作为 Stage 1.2 第一版主归一化。

### 4.4 GPU / tensor-first

正式 imageization 函数应以 tensor 为主：

```text
input:  torch.Tensor [B, L]
period: torch.Tensor 或 list[int]，长度 B
output: torch.Tensor [B, V, H, W]
```

实现原则：

- 优先用 `torch` 张量操作。
- 支持 CPU device smoke，但接口设计必须可直接迁移到 GPU。
- 不以 matplotlib/PIL 作为正式路径。
- PNG 只从少量 sampled tensor 后处理得到，用于人工检查。
- 记录 imageization latency，包括 CPU smoke latency；如果 GPU 可用，后续实现应记录 GPU latency。

## 5. 三视图协议

### 5.1 View 0：`line_raster`

目的：

> 近似线图 / 时序轮廓，保留时间顺序和局部形状。

输入：

```text
x_norm: [B, L]
```

输出：

```text
line_raster: [B, H, W]
```

第一版建议：

- 时间轴从 `L` 线性重采样到 `W`。
- 数值轴从 `[clip_min, clip_max]` 映射到 `[0, H-1]`。
- 使用 tensor soft raster，而不是 matplotlib line plot。
- 每个时间位置在高度轴上放一个 soft point，后续可扩展为连接相邻点的 soft line。

默认尺寸：

```text
H = 64
W = 192
```

说明：

- `W=192` 对齐当前 `history_len=192`，避免第一版引入额外 resize 误差。
- 后续如果接标准 ViT，可在 encoder adapter 中处理 resize，而不是在 Stage 1.2 改协议。

### 5.2 View 1：`period_fold`

目的：

> 借鉴 TimesNet / VisionTS 的 period reshape，将 1D history 按 period 折叠成 2D 周期结构。

输入：

```text
x_norm: [B, L]
period: hour=24, min=144
```

输出：

```text
period_fold: [B, H, W]
```

第一版规则：

1. 对每个样本，根据 registry 中的 `period` 取 fold width。
2. 将 `x_norm` padding 到 `ceil(L / period) * period`。
3. reshape 为 `[num_cycles, period]`。
4. 将该 2D 矩阵 resize / pad 到固定 `[H, W]`。
5. padding 位置使用 0，并在 manifest 中记录 padding 策略。

默认：

```text
hour period = 24
min period = 144
history_len = 192
target tensor size = [64, 192]
```

对 `hour`：

```text
192 = 8 * 24
num_cycles = 8
```

对 `min`：

```text
192 = 1 * 144 + 48
padding 后 num_cycles = 2
```

`min` 的 period_fold 会包含 padding；这是可接受的，但必须记录在 `image_index.csv` 或 manifest 中，便于后续解释。

### 5.3 View 2：`fft_power`

目的：

> 提供轻量频域视图，补充 line_raster 和 period_fold 对频率结构的表达。

第一版规则：

- 对 `x_norm` 做 `torch.fft.rfft`。
- 取 power spectrum。
- 去掉或弱化 DC 分量。
- 对 power 做 `log1p` 压缩。
- 归一化到 `[0, 1]`。
- 将 1D frequency power 扩展为 `[H, W]`，例如沿高度方向 repeat 或生成 frequency-raster。

注意：

- `fft_power` 是轻量视图，不等价于 STFT/CWT。
- 本阶段不实现多尺度 spectrogram。
- 如果后续发现该 view 信息不足，再作为 Stage 1.2b 或 Stage 1.3 消融扩展。

## 6. Smoke 抽样策略

Stage 1.2 不直接处理 627,430 条全量窗口。第一版 smoke 推荐从正式 working registry 抽样：

```text
每个 subset x split x official_tsf_cell 抽取最多 N 条
默认 N = 4 或 8
```

推荐默认：

```text
max_per_group = 8
group_cols = subset, split, official_tsf_cell
random_seed = 20260607
```

理论最大规模：

```text
2 subsets * 3 splits * 8 cells * 8 = 384 windows
```

如果某些 group 不足，则保留实际可用数量。

该抽样只用于协议 smoke，不定义新的正式 sample_set。输出目录应带 smoke 后缀，避免覆盖未来正式 image tensor 缓存。

## 7. 输出协议

建议输出目录：

```text
outputs/vision_ts_routing/image_tensors/
  qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1/
```

建议文件：

```text
view_tensor_sample.npz
image_index.csv
manifest.json
debug_png/
```

其中：

- `view_tensor_sample.npz`：保存 smoke tensor，shape `[N, V, H, W]`，第一版 `V=3`。
- `image_index.csv`：每行对应 tensor 中一个样本，保留 `physical_window_id`、`sample_set_id`、`subset`、`split`、`item_id`、`channel`、`period`、normalization metadata、padding metadata。
- `manifest.json`：记录协议版本、视图定义、shape、抽样策略、输入 registry/proxy 路径、latency。
- `debug_png/`：少量 PNG，用于人工检查，不作为训练输入。

`image_index.csv` 必须可以与 Stage 1.1 proxy 按 `physical_window_id` join。

## 8. Manifest 必备字段

```json
{
  "stage": "stage1_2_imageization_protocol_smoke",
  "image_protocol_id": "image3_h64_w192_v1",
  "sample_set_id": "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e",
  "base_registry_id": "qb_h192_p96_quito_overlap_8478f330",
  "input_registry_dir": "...",
  "input_proxy_dir": "...",
  "view_names": ["line_raster", "period_fold", "fft_power"],
  "view_dim": 3,
  "tensor_shape": ["N", "V", 64, 192],
  "view_axis_semantics": {
    "line_raster": {"x": "time_index", "y": "normalized_value_height"},
    "period_fold": {"x": "phase_within_period", "y": "cycle_or_period_block"},
    "fft_power": {"x": "frequency_bin", "y": "rasterized_frequency_power"}
  },
  "normalization": {
    "method": "instance_mean_std",
    "scope": "per_physical_window_id_history",
    "norm_const": 0.4,
    "eps": 1e-5,
    "clip_min": -5.0,
    "clip_max": 5.0,
    "future_read_policy": "history_only"
  },
  "sampling": {
    "group_cols": ["subset", "split", "official_tsf_cell"],
    "max_per_group": 8,
    "random_seed": 20260607
  },
  "channel_policy": "sample_channel_independent",
  "view_tensor_semantics": "multi_view_not_rgb",
  "debug_png_policy": "sampled_only",
  "runs_visual_encoder": false,
  "runs_expert_models": false,
  "implements_router": false
}
```

## 9. 测试要求

Stage 1.2 实现前应先写测试。

最低测试覆盖：

1. `normalize_history()` 只基于 history 输入，输出有限值，并返回 metadata。
2. `imageize_batch()` 接受 `[B, L]` 和 period，输出 `[B, V, H, W]`，第一版 `V=3`。
3. `period_fold` 对 `period=24` 的 192 点序列不需要 padding。
4. `period_fold` 对 `period=144` 的 192 点序列需要 padding，并记录 padding 信息。
5. `physical_window_id` 在 `image_index.csv` 中唯一。
6. smoke 输出目录不会覆盖未来正式输出。
7. debug PNG 生成只依赖 tensor 后处理，不是正式 imageization 路径。
8. 与 Stage 1.1 proxy 可按 `physical_window_id` join。

验证命令建议：

```bash
conda run -n quito python -m pytest tests/test_quitobench_imageization_protocol.py -q
conda run -n quito python tools/quitobench_imageization_protocol.py --smoke
conda run -n quito python -m pytest tests -q
```

## 10. 延后项

以下问题重要，但不在 Stage 1.2 第一版实现：

- 是否直接使用 frozen ImageNet ViT。
- 是否为 view tensor 增加 learned 1x1 adapter。
- 是否改为 per-view grayscale repeat 后分别过视觉 encoder。
- 是否加入 GAF / RP / CWT / STFT。
- 是否做 multichannel imageization。
- 是否保存全量 627,430 条 tensor 缓存。
- 是否抽取 visual embedding。
- 是否训练 visual router。

这些内容应在 Stage 1.3 或 Stage 1.2b 中作为消融或 encoder 适配问题处理。

## 11. 风险与检查点

### 风险 1：3 个 view 被误当作 RGB

处理：

- 代码、manifest、文档统一称为 `view_tensor` 或 `view_dim`。
- debug 默认三联灰度图。
- 后续接 frozen RGB ViT 时必须单独记录 adapter 策略。

### 风险 2：图像化引入伪迹

处理：

- Stage 1.2 只做 smoke，不直接报告模型效果。
- debug PNG 覆盖不同 subset、split、official_tsf_cell。
- 后续 Stage 1.3 必须和 Stage 1.1 proxy、普通时序 encoder 做对照。

### 风险 3：在线路径过慢

处理：

- 主实现使用 torch tensor 操作。
- 记录 batch latency。
- matplotlib/PIL 只用于 sampled debug 输出。

### 风险 4：归一化泄漏 future

处理：

- imageization 函数只接收 history tensor，不接收 target。
- 从 registry 切片时只读取 `[history_start_idx, history_end_idx)`。
- 测试中构造 future 极值，确认输出不受 target 影响。

## 12. Stage 1.2 完成标准

Stage 1.2 完成时应具备：

- 中文实验日志。
- `tools/quitobench_imageization_protocol.py`。
- `tests/test_quitobench_imageization_protocol.py`。
- smoke tensor 输出。
- `image_index.csv` 与 `manifest.json`。
- 少量 debug PNG。
- 全量测试通过。
- 明确记录没有训练 encoder、没有实现 router、没有运行专家模型。

完成后再进入 Stage 1.3：视觉 encoder 表征 / 预训练设计与 smoke。
