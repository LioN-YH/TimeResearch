# Stage 1.1b：torch light proxy online kernel

## 1. 实验目的

为 Stage 1.1 light proxy 增加在线可用的 torch batch kernel：

```python
compute_light_proxy_torch(histories: torch.Tensor, periods: torch.Tensor) -> torch.Tensor
```

本阶段目标不是重算 Stage 1.1 proxy cache，而是补充在线路径：

- 支持 `[B, L]` history batch。
- 支持 CPU/GPU device。
- 不依赖 Pandas。
- 不读取 future target。
- 输出 feature order 与 Stage 1.1 `FEATURE_COLUMNS` / manifest 一致。
- 数值上与当前 NumPy reference `compute_window_proxy()` 对齐。

## 2. 实验计划

1. 在 `docs/superpowers/plans/` 写 Stage 1.1b 实现计划。
2. 按 TDD 在 `tests/test_quitobench_sample_channel_light_proxy.py` 中新增 torch kernel 测试。
3. 在 `tools/quitobench_sample_channel_light_proxy.py` 中新增 `compute_light_proxy_torch()`。
4. 验证 CPU toy data 与 NumPy reference 一致。
5. CUDA 可用时验证 GPU 输出保留在 CUDA，并与 CPU 输出一致。
6. 跑全量测试。
7. 写实验日志并更新总览。

## 3. 执行命令

红灯测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
```

预期失败：

```text
ImportError: cannot import name 'compute_light_proxy_torch'
```

实现后 Stage 1.1 测试：

```bash
conda run -n quito python -m pytest tests/test_quitobench_sample_channel_light_proxy.py -q
```

结果：

```text
7 passed in 3.09s
```

全量测试：

```bash
conda run -n quito python -m pytest tests -q
```

结果：

```text
38 passed in 3.06s
```

CPU/GPU parity 验证：

```bash
conda run -n quito python -c "<构造 toy histories/periods，分别调用 CPU 和 CUDA compute_light_proxy_torch>"
```

结果：

```text
cuda_available True
feature_count 15
cpu_shape (2, 15) cpu
cpu_finite True
gpu_shape (2, 15) cuda:0
gpu_finite True
max_abs_cpu_gpu 1.1920928955078125e-07
```

## 4. 输入数据与配置

本阶段不读取 registry、parquet 或 Stage 1.1 cache。

测试输入为 toy history batch：

```text
histories shape = [2, 6]
periods shape = [2]
```

输出 feature order 来自：

```python
FEATURE_COLUMNS = [
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "amplitude",
    "last_value",
    "missing_ratio",
    "slope",
    "recent_std_ratio",
    "acf_lag1",
    "acf_period",
    "spectral_entropy",
    "dominant_frequency_strength",
]
```

## 5. 实验结果

代码变更：

- `tools/quitobench_sample_channel_light_proxy.py`
  - 新增 torch import。
  - 新增 `compute_light_proxy_torch(histories, periods)`。
  - 新增内部 torch helper，用于 finite quantile、last finite、slope、autocorr。
- `tests/test_quitobench_sample_channel_light_proxy.py`
  - 新增 CPU parity 测试。
  - 新增 CUDA 可用时的 device / CPU-GPU 一致性测试。
- `docs/superpowers/plans/2026-06-07-stage1-1b-torch-light-proxy.md`
  - 记录 Stage 1.1b 实现计划。

验证结果：

- CPU toy data 与 NumPy `compute_window_proxy()` 在 `rtol=1e-5, atol=1e-5` 下对齐。
- CUDA 可用时，输出 device 为 `cuda`。
- CPU/GPU 最大绝对误差约 `1.19e-07`。
- 输出 shape 为 `[B, 15]`，15 与 `FEATURE_COLUMNS` 一致。

## 6. 问题与观察

- 当前 torch kernel 保留了 Stage 1.1 reference 的语义，包括：
  - missing value 以 finite mean 填充后计算 ACF、recent ratio 和频域特征；
  - `std`、`mean`、`median`、`iqr` 等统计只基于 finite values；
  - `last_value` 使用最后一个 finite value；
  - `acf_period` 支持每个样本不同 period。
- 第一版 torch kernel 为了忠实对齐 reference，在 quantile、slope 和 variable period ACF 上仍有 per-row loop。它已经支持 CPU/GPU device，但还不是最终高吞吐优化版。
- 如果后续需要高 QPS 在线推理，可继续做 Stage 1.1c：进一步向量化 quantile/variable-lag ACF，或按固定 period 分组 batch。

## 7. 结论

Stage 1.1b 已完成。

当前项目同时具备：

- Stage 1.1 NumPy/Pandas reference + 离线 cache 生成脚本；
- Stage 1.1b torch batch kernel，用于在线 CPU/GPU 路径。

后续 Stage 1.3 可以在视觉 encoder pipeline 中直接调用 `compute_light_proxy_torch()`，避免在线路径依赖 Pandas。

## 8. 下一步计划

1. Stage 1.3 设计视觉 encoder 输入消费策略时，将 `compute_light_proxy_torch()` 作为在线 proxy kernel。
2. 若后续进行 latency benchmark，应同时报告：
   - view tensor imageization latency；
   - torch proxy latency；
   - visual encoder forward latency；
   - proxy + visual end-to-end latency。
3. 当前仍不要实现 router，不运行专家模型。
