# Stage 1.2b：Proxy + Imageization 在线 Latency Sweep

## 1. 实验目的

系统测量 Stage 1.1b torch light proxy kernel 与 Stage 1.2 三视图 `view_tensor` imageization 在不同 batch size、CPU/GPU 下的在线计算成本，为 Stage 1.3 视觉 encoder adapter smoke 前的在线路径预算提供依据。

本实验明确不实现 router、不运行专家模型、不训练或运行视觉 encoder、不重新计算 Stage 1.1 离线 proxy cache。

## 2. 实验计划

- 复用正式 Stage 1.0 working registry：
  - `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/`
- 从 registry 抽取 1024 条 sample-channel history-only window，覆盖 `hour/min`。
- 对 `batch_size = 1, 8, 32, 128, 512, 1024` 分别测量：
  - `proxy_torch_latency_ms_per_window`
  - `view_tensor_latency_ms_per_window`
  - `proxy_plus_view_latency_ms_per_window`
- 对 `device = cpu, cuda` 分别计时。
- 每个配置使用 `warmup_iters=3`、`measure_iters=10`。

## 3. 执行命令

```bash
conda run -n quito python -m pytest tests/test_quitobench_proxy_imageization_latency.py -q
conda run -n quito python -m pytest tests/test_quitobench_proxy_imageization_latency.py tests/test_quitobench_sample_channel_light_proxy.py tests/test_quitobench_imageization_protocol.py -q
conda run -n quito python tools/quitobench_proxy_imageization_latency.py
```

输出校验命令检查了 CSV 行数、CPU/CUDA 覆盖、三个 latency 指标非空且非负，以及 manifest 中 `runs_expert_models=false`、`implements_router=false`、`recomputes_stage1_1_cache=false`。

## 4. 输入数据与配置

- Conda 环境：`quito`
- Registry：`qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e`
- 抽样窗口数：1024
- History length：192
- View tensor：`[B, 3, 64, 192]`
- Proxy feature count：15
- CUDA available：`true`
- 输出：
  - `outputs/vision_ts_routing/latency/stage1_2b_proxy_imageization_latency.csv`
  - `outputs/vision_ts_routing/latency/stage1_2b_proxy_imageization_latency_manifest.json`

## 5. 实验结果

| device | batch_size | proxy ms/window | view tensor ms/window | proxy + view ms/window |
| --- | ---: | ---: | ---: | ---: |
| cpu | 1 | 1.143483 | 0.648323 | 1.526906 |
| cpu | 8 | 0.391602 | 0.881494 | 1.553349 |
| cpu | 32 | 0.512532 | 0.362014 | 0.908024 |
| cpu | 128 | 0.330087 | 0.237669 | 0.581267 |
| cpu | 512 | 0.283286 | 0.212843 | 0.500413 |
| cpu | 1024 | 0.286314 | 0.236161 | 0.540026 |
| cuda | 1 | 1.855508 | 0.877756 | 2.728721 |
| cuda | 8 | 0.849088 | 0.169590 | 1.027060 |
| cuda | 32 | 0.750594 | 0.091549 | 0.841922 |
| cuda | 128 | 0.731576 | 0.073452 | 0.799438 |
| cuda | 512 | 0.723430 | 0.069253 | 0.803487 |
| cuda | 1024 | 0.745576 | 0.083346 | 0.835049 |

测试结果：

- `tests/test_quitobench_proxy_imageization_latency.py`：2 passed
- 相关回归测试：13 passed
- 输出校验：`manifest_ok 1024 True`

## 6. 问题与观察

- CUDA 上 `view_tensor` 单独计算在 batch >= 8 后明显快于 CPU。
- CUDA 上 `proxy_torch` 当前没有表现出优势，batch >= 128 时约 `0.72-0.75 ms/window`，高于 CPU 的约 `0.28-0.33 ms/window`。这符合当前 Stage 1.1b kernel 中部分 per-row 逻辑和标量同步仍偏 Python/row-loop 的实现特征。
- CPU 上 `proxy + view` 在大 batch 下约 `0.50-0.54 ms/window`；CUDA 上大 batch 约 `0.80-0.84 ms/window`，主要被 proxy kernel 拖慢。
- 本实验只测 proxy 和 imageization，不包含 Stage 1.3 视觉 encoder forward latency。

## 7. 结论

Stage 1.2b latency sweep 已完成。当前在线路径中，`view_tensor` imageization 可从 GPU batch 化获益，但 `compute_light_proxy_torch()` 仍是大 batch CUDA 路径的主要瓶颈。进入 Stage 1.3 前，可以先接受该成本用于 smoke；若后续在线预算要求严格，应单独优化 proxy kernel 的 row-loop/quantile/autocorr 实现。

## 8. 下一步计划

- 进入 Stage 1.3a：visual encoder adapter smoke。
- Stage 1.3a 需要把本次 `view_tensor` latency 与 encoder forward latency 分开报告。
- 不要在 Stage 1.3a 直接实现 router；router/gate 仍等待视觉 embedding 和专家预测缓存接口稳定。
