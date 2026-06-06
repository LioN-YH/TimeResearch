# 视觉模型迁移到时序建模：批判性修订版

## 1. 修订后的核心命题

本课题不应停留在“把 ViT 接到时序预测模型前面”，而应聚焦：

> 经过 TSF 结构语义微调的视觉编码器，能否在 few-shot 场景下从 GPU 端内生伪图像张量中提取跨 regime 的通用结构先验，并据此给出模型选择或连续融合权重。

这里的关键变化有四个：

1. 主数据集从传统 ETT/Electricity/Traffic 等转向 QuitoBench。
2. “时序图像”定义为多维张量，不定义为落盘图片。
3. ViT 必须做时序语义微调，冻结视觉特征只作为 baseline。
4. 路由从 VisMoE 式离散路由升级为连续路由。

## 2. 关于 QuitoBench：可以直接作为主实验数据

你的判断是成立的。传统长时预测数据集常按数据源划分，例如 ETT、Electricity、Traffic、Weather、Exchange、ILI，但这些数据集内部的 trend、seasonality、forecastability 分布不均衡。若只报整体 MSE/MAE，很可能掩盖模型在不同结构 regime 上的行为差异。

QuitoBench 更适合本课题，因为它直接围绕 TSF 结构体系组织数据：

- trend
- seasonality
- forecastability
- 八个 trend × seasonality × forecastability 组合 cell

本地 Quito 仓库也支持这一点：

- `README.md` 明确列出 forecastability、seasonality strength、stationarity、missing data、variability 等质量指标。
- `dataset_quality.py` 提供 `forecastability_welch`、`stl_decomposition_strengths`、`trend_strength_stl`、`seasonality_strength_stl` 等实现。
- Benchmark protocol 中包含 quality stratification 和 zero-shot evaluation。

因此建议改成：

### 主实验

- QuitoBench hour config
- QuitoBench min config
- 每个 TSF cell 单独报告结果
- 汇总结果必须同时给 macro-average 和 weighted-average

### 外部验证

传统数据集仍保留，但不作为主结论依据：

- ETTm1/ETTm2/ETTh1/ETTh2
- Electricity
- Traffic
- Weather
- Exchange
- ILI

它们的角色是 sanity check：确认模型不是只拟合 Quito/CloudOps 分布。

## 3. 关于伪图像：应明确为 GPU 张量而非真实图片

你对在线场景的担心非常关键。很多时序图像化库，例如基于 NumPy/scipy/pyts 的 GAF、RP、MTF、CWT 实现，默认 CPU 计算。如果在线推理路径是：

`GPU time series -> CPU imageization -> GPU ViT`

那么 CPU/GPU 往返会吞掉视觉模块的收益，尤其在小 batch、低延迟场景中。

所以方法定义应改为：

> 内生伪图像是用于 ViT 的多维张量表示 `I in R^{B x V x H x W}` 或 `I in R^{B x C_img x H x W}`，真实图片保存只用于 debug 和可解释性检查。

建议实现两条路径：

1. **训练/在线路径：GPU tensor imageization**
   用 PyTorch 张量算子实现周期重排、GAF、FFT/STFT power map，尽量避免 CPU 中间态。

2. **检查路径：save_image/debug_plot**
   抽样把伪图像张量保存为 PNG，用于人工检查结构是否合理。

MVP 伪图像建议：

- 周期重排 2D：最便宜，直接 reshape/pad/index。
- GASF/GADF：可用矩阵广播在 GPU 上实现。
- FFT/STFT power map：比 CWT 更容易 GPU batch 化。

CWT 和复杂 RP/MTF 暂缓到第二阶段。它们不是不能做，而是容易把课题拖到“高成本图像化工程”上。

## 4. 关于 few-shot：ViT 微调是必要的

如果目标是 TimeFuse 式“每个数据集适配后再集成”，视觉模块的贡献会被削弱，因为下游专家已经充分适配数据集。你的目标更有价值：

> 在 few-shot 目标域中，视觉模块利用通用结构先验，快速判断哪些专家更适合当前样本或当前 regime。

因此实验应区分三层能力：

1. **专家适配能力**
   专家在源数据/源 regime 上训练或微调。

2. **视觉结构先验能力**
   ViT 在大规模窗口上通过 TSF 标签、教师表征、重建任务进行微调。

3. **few-shot 路由校准能力**
   目标 regime 只给少量样本，校准 router、温度或最后一层。

建议训练目标：

`L = L_tsf + lambda_route * L_route + lambda_align * L_align + lambda_rec * L_rec`

其中：

- `L_tsf`：预测 trend、seasonality、forecastability 和 8-cell regime。
- `L_route`：根据专家 oracle error 分布学习连续权重。
- `L_align`：对齐强时序模型中间表征。
- `L_rec`：mask 后重建原始标准化序列或伪图像 patch。

第一阶段建议只做：

`L_tsf + L_route`

这样能直接验证“结构语义是否帮助路由”。`L_align/L_rec` 放到第二阶段，否则变量太多。

## 5. 关于 VisMoE：应作为强相关 baseline，而不是简单复现

VisMoE 和你的思路确实接近：它把序列画成 line chart，利用 VLM/视觉模块识别 temporal regime，再做离散路由。这个方向可以借鉴，但本课题要避免只变成 VisMoE 的小改版。

建议差异点写清楚：

1. **输入不同**
   VisMoE 使用真实 line-chart image；本课题使用 GPU 端内生伪图像张量。

2. **路由不同**
   VisMoE 偏离散 regime routing；本课题主方法是连续 soft/top-k routing。

3. **训练目标不同**
   本课题显式微调 ViT 学 TSF 结构语义和专家误差分布。

4. **应用场景不同**
   本课题强调 few-shot 模型选择/融合，不只是标准监督预测。

连续路由建议：

`w = softmax(g(z_v) / tau)`

或：

`w = entmax(g(z_v) / tau)`

其中：

- `tau` 控制离散化程度。
- `softmax` 稳定，适合作主方法。
- `entmax/sparsemax` 可得到稀疏权重，适合减少推理成本。

需要同时保留三个路由 baseline：

- hard top-1 routing：对齐 VisMoE。
- soft top-k routing：主方法。
- dense soft routing：分析上界。

## 6. 修订后的模型框架

输入：

`X in R^{B x L x C}`

GPU 伪图像化：

`I = Phi_gpu(X) in R^{B x V x H x W}`

视觉结构编码：

`z_v = ViT(I)`

连续专家路由：

`w = Router(z_v, optional_stats)`

专家预测：

`y_k = Expert_k(X)`

融合：

`y_hat = sum_k w_k * y_k`

专家池建议：

- seasonal naive：低成本强 baseline。
- DLinear：趋势/季节分解专家。
- PatchTST：patch transformer 专家。
- TSMixer/TimeMixer：MLP/multiscale 专家。
- iTransformer：多变量阶段加入。

## 7. 修订后的实验设计

### 7.1 主实验：QuitoBench TSF 分层

每个 TSF cell 单独报告：

- MSE
- MAE
- MASE
- SMAPE
- oracle gap
- expert utilization entropy
- routing entropy
- routing-regime agreement

整体结果报告：

- macro average over 8 TSF cells
- weighted average over samples

### 7.2 Few-shot 设置

目标 regime 给：

- 1-shot
- 5-shot
- 10-shot
- 50-shot

可适配部分：

- 只校准 router temperature
- 只训练 router last layer
- adapter/LoRA 微调 ViT
- 专家也 few-shot 微调

最重要的对照是：

> 同样 few-shot 预算下，是调专家更有效，还是调视觉 router 更有效。

### 7.3 跨 regime 泛化

使用 leave-one-cell-out：

- 在 7 个 TSF cell 上训练视觉 router。
- 在剩余 1 个 cell 上测试。
- 轮换 8 次。

这个实验能直接回答“视觉模块是否学到跨结构泛化能力”。

### 7.4 对照组

必须包含：

- 单专家 best fixed expert
- 均匀 ensemble
- 验证集全局固定权重 ensemble
- TimeFuse 风格统计 meta-feature router
- 时序 encoder router
- VisMoE 风格 hard routing
- TSF 8-class hard routing
- 视觉连续 soft/top-k routing
- 视觉 + 统计联合 routing

## 8. MVP 建议

第一版建议控制变量：

1. 数据：QuitoBench hour/min，按 8 个 TSF cell 分层。
2. 伪图像：GPU 周期重排、GAF、FFT/STFT power map。
3. ViT：ImageNet/MAE 初始化，先冻结，再 adapter/LoRA。
4. 专家：seasonal naive、DLinear、PatchTST、TSMixer。
5. 路由：softmax continuous routing。
6. 监督：`L_tsf + L_route`。
7. few-shot：1/5/10/50-shot 校准 router。
8. baseline：统计 router、hard TSF router、VisMoE 风格 line-chart hard router。

MVP 的成功标准不是只看整体 MSE/MAE，而是：

- 视觉连续路由是否降低 oracle gap。
- 是否在低样本目标 regime 中优于统计 router。
- 是否在 8 个 TSF cell 上更均衡，而不是只提升某一类。
- GPU 伪图像化是否避免明显在线延迟瓶颈。

## 9. 需要进一步一起定的点

我认为目前还有四个点值得停下来讨论：

1. **专家池规模**
   专家太多会增加 oracle label 成本，太少又体现不出路由价值。MVP 建议 4 个专家。

2. **few-shot 的对象**
   是对新数据集 few-shot，还是对 QuitoBench 的 held-out TSF cell few-shot？建议先做后者。

3. **ViT 输入尺寸**
   如果强行对齐 224x224，会有插值和成本问题。可以考虑小尺寸 ViT 或 patch embedding 直接吃 `H x W` 张量。

4. **是否把 GPU imageization 写成贡献**
   如果后续实现质量足够好，它可以成为方法和系统层面的双重贡献。


## 10. 对当前讨论的补充

### 10.1 held-out TSF cell 是什么

QuitoBench 的 TSF cell 指 trend、seasonality、forecastability 三个二值/分层属性组成的 8 个结构格子：

`2 trend levels x 2 seasonality levels x 2 forecastability levels = 8 cells`


held-out TSF cell 的意思是：

1. 训练时拿其中 7 个 cell。
2. 完全不使用剩下 1 个 cell 的训练样本。
3. 测试时只在这个被留出的 cell 上评估。
4. 轮换 8 次，每次留出不同 cell。

这个设置不是为了模拟真实部署的全部情况，而是为了强行检验：

> ViT/router 是否学到了可迁移结构先验，而不是记住某个 regime 的专家偏好。

不过它难度较高，MVP 可以先做弱一点的版本：

- 每个 cell 内部 train/valid/test。
- 每个 cell 单独报告专家偏好和 router 性能。
- 再做 leave-one-cell-out 作为泛化实验。

### 10.2 小尺寸 patch embedding 的建议

如果性能影响不大，我建议采用小尺寸 patch embedding 直接吃 `H x W` 伪图像张量，而不是强行 resize 到 224 x 224。

理由：

1. TimesNet 已经说明 1D 序列可被重排为 2D tensor，并直接在 2D tensor 上建模，不需要真实图片尺寸。
2. ViT 的本质是 patchify + linear projection + positional embedding。只要 patch 数和位置编码处理合理，输入不必固定为 224 x 224。
3. VisionTS 使用视觉 MAE 说明视觉先验有价值，但我们的目标是结构先验路由，不需要完全继承 224 x 224 的图像设定。

建议做三个版本对照：

| 输入方案 | 优点 | 风险 |
| --- | --- | --- |
| resize 到 224 x 224 + 预训练 ViT | 最大程度利用现成 ViT/MAE | 插值可能扭曲时序结构，计算成本较高 |
| 小尺寸 ViT，从预训练迁移 patch embedding/部分层 | 成本低，适配伪图像 | 需要处理 positional embedding |
| 自定义 patch embedding + ViT encoder | 最贴合 `H x W` 张量 | 预训练利用弱，训练数据需求更高 |

MVP 推荐：

`H x W` 直接 patchify，patch size 取 `4/8/16` 做消融；位置编码用 2D learnable PE 或插值 PE。

### 10.3 专家池不应拍脑袋选，应先做 cell-level profiling

你的建议更稳妥：先在 QuitoBench 上测试代表性专家，拉出每个 TSF cell 的表现，再决定专家池。

建议流程：

1. 选 6-8 个候选专家。
2. 在 QuitoBench 每个 cell 上统一训练/微调。
3. 输出 `cell x model` 性能矩阵。
4. 计算每个模型在不同 cell 的 rank、win rate、oracle contribution。
5. 只保留互补性强的专家进入 MoE。

候选专家建议覆盖 TimeRecipe 的模块维度：

- seasonal naive：统计/季节基线。
- DLinear：decomposition + linear。
- FITS：frequency embedding + MLP。
- PatchTST：patch embedding + Transformer。
- iTransformer：feature/invert embedding + Transformer。
- TSMixer/TimeMixer：MLP/multiscale。
- Crossformer：cross-dimension dependency，多变量阶段可用。

选择专家不看单模型平均分，而看互补性：

- 如果两个模型在 8 个 cell 的 rank 几乎一致，只保留更快/更稳的。
- 如果某模型平均分一般，但在低 forecastability 或强 seasonality cell 明显领先，应保留。
- 如果某模型对 context length 特别敏感，也应保留，因为 QuitoBench 论文提到 context length 会影响模型族优势。

### 10.4 借鉴 TimeRecipe 的消融方式

TimeRecipe 的价值在于 module-level benchmarking。我们可以把它迁移成：

`伪图像模块 x 时序特征模块 x 路由模块 x 专家模块`

建议模块表：

| 维度 | 选项 |
| --- | --- |
| 伪图像 | period reshape, GAF, FFT/STFT, RP, MTF, CWT |
| 时序特征 | TSF labels, catch22/stat features, ACF/PACF, spectral entropy |
| 视觉编码 | frozen ViT, LoRA ViT, small ViT, from-scratch ViT |
| 路由 | hard top-1, softmax, sparsemax/entmax, top-k |
| 专家 | naive, DLinear, FITS, PatchTST, iTransformer, TSMixer/TimeMixer |

实验不要穷举全部组合。先做两阶段：

1. **profiling 阶段**
   只评估专家池和伪图像单视图，找到互补结构。

2. **recipe 阶段**
   固定专家池，系统评估伪图像组合、视觉微调目标和路由方式。

最终论文可以形成一个类似 TimeRecipe 的结论：

> 哪些伪图像/视觉路由设计适合哪些 TSF regime。

TimeRecipe: https://openreview.net/forum?id=CsoR8ztROC

## 11. 参考链接

- QuitoBench paper: https://arxiv.org/abs/2603.26017
- QuitoBench GitHub: https://github.com/alipay/quito
- QuitoBench dataset: https://huggingface.co/datasets/alipay/QuitoBench
- TimeFuse: https://arxiv.org/abs/2505.18442
- VisMoE CIKM 2025 TOC: https://www.sigweb.hosting.acm.org/toc/cikm25.html
- VisionTS: https://arxiv.org/abs/2408.17253
- TimesNet: https://openreview.net/forum?id=ju_Uqw384Oq
- TimeRecipe: https://openreview.net/forum?id=CsoR8ztROC
