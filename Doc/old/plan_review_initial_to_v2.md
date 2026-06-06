# 从初始构想到 v2 方案的审阅

## 1. 总体判断

从初始版到 v2，方案的主线变清晰了：

初始版的核心是：

> ViT 作为全局结构先验提取器，通过内生伪图像为多时序专家做路由。

v2 的核心被收窄为：

> 面向 few-shot 场景，微调后的视觉编码器从 GPU 端伪图像张量中学习 TSF 结构先验，并输出连续专家融合权重。

这个收窄是正确的。它把课题从“视觉模型能不能做时序”变成“视觉结构先验能否帮助 few-shot 模型选择/融合”，可验证性更强，也更容易和 VisionTS、TimeFuse、VisMoE 区分。

## 2. 已经变强的部分

### 2.1 功能定位更清楚

初始版提出 ViT 是“全局结构先验提取器”，但还没有说明如何证明这个定位。

v2 增加了：

- TSF 标签监督。
- 专家 oracle error 分布监督。
- cell-level reporting。
- oracle gap。
- routing entropy。
- routing-regime agreement。

这些指标让“功能定位”从概念变成了可检验对象。

### 2.2 数据集选择更贴合问题

第一版方案仍把 ETT/Electricity/Traffic/Weather 等放在第一阶段。这个选择有风险，因为这些数据集的结构分布不均，会让整体误差掩盖 regime 差异。

v2 改成 QuitoBench 作为主数据集更合理，因为本课题本质上要学习 trend、seasonality、forecastability 结构先验。

建议最终写法：

> QuitoBench 是主实验数据；传统 LTSF 数据集是外部泛化检查，不承担主结论。

### 2.3 “伪图像”定义更工程化

初始版容易被理解成真实图片路径：序列转 PNG/JPEG 再喂 ViT。

v2 明确为 GPU tensor imageization，这是必要修正。否则在线场景会被 CPU/GPU 往返拖垮。

建议继续强化：

- 训练/推理路径只允许 tensor。
- 图片保存只用于 debug。
- 报告 GPU imageization latency。

### 2.4 路由从离散升级为连续是合理创新点

VisMoE 已经覆盖了“视觉识别 regime + 离散路由”的思路。

v2 用 continuous routing 区分开来，是更稳的创新方向：

- softmax routing 作为主方法。
- top-k/entmax 作为稀疏版本。
- hard top-1 作为 VisMoE 对照。

## 3. 仍然薄弱或需要收敛的部分

### 3.1 few-shot 场景定义还不够精确

目前 v2 里有两种 few-shot 语义：

1. 新数据集 few-shot。
2. QuitoBench held-out 或 low-shot TSF cell。

这两者难度和实验含义不同。建议先选一个作为主线。

推荐主线：

> QuitoBench cell-level low-shot router adaptation。

即每个 TSF cell 内只给少量样本校准 router，而不是一开始做完全 held-out cell。

held-out cell 可以作为泛化压力测试，不作为 MVP 主实验。

### 3.2 ViT 微调目标仍可能过多

初始版提出标签预测、表征对齐、重建三个损失。v2 又加入 route loss 和 TSF loss。

这会带来一个风险：如果最终有效，很难解释是哪一个损失起作用。

建议分阶段：

1. `L_tsf`：验证 ViT 是否能识别结构。
2. `L_route`：验证结构是否能转化为专家偏好。
3. `L_tsf + L_route`：主方法。
4. `+ L_align`：增强实验。
5. `+ L_rec`：最后加入，除非重建任务被明确证明有用。

### 3.3 专家池需要先实证筛选

v2 已经加入 cell-level profiling，这是必须的。否则专家池会变成主观选择。

建议先不要固定 DLinear/PatchTST/TSMixer 为最终专家，而是固定候选集：

- seasonal naive
- DLinear
- FITS
- PatchTST
- iTransformer
- TSMixer/TimeMixer
- Crossformer

用 `cell x model` 矩阵决定最终 3-4 个专家。

筛选标准应包括：

- average rank
- cell-wise win rate
- oracle ensemble contribution
- latency
- parameter count
- few-shot adaptation cost

### 3.4 小尺寸 patch embedding 需要做成消融而非默认结论

v2 倾向直接吃 `H x W` 张量是合理的，但不能直接假设它性能不掉。

建议把输入设计作为明确消融：

1. 224 resize + pretrained ViT。
2. `H x W` small ViT + interpolated positional embedding。
3. `H x W` custom patch embedding + ViT encoder。

如果第 2 种性能接近第 1 种且延迟更低，就可以成为主方法。

### 3.5 RAG 和 Prototype Bank 应继续后置

初始版有 RAG 扩展，第一版文档也保留了 prototype/RAG。

目前主线已经包含：

- QuitoBench TSF 分层。
- GPU 伪图像。
- ViT 微调。
- 连续路由。
- 专家池 profiling。
- few-shot adaptation。

变量已经足够多。RAG/Prototype Bank 建议只放未来工作，不进入第一篇主实验。

## 4. 建议的最终主线

建议最终方案写成三阶段。

### Stage A: Expert Profiling

目的：

> 找出 QuitoBench 8 个 TSF cell 中，不同专家的偏好和互补性。

输出：

- `cell x model` 性能矩阵。
- 每个专家的 win rate。
- oracle ensemble 上界。
- 最终专家池。

### Stage B: Visual Structure Pretraining

目的：

> 让 ViT 学会识别 TSF 结构，而不是直接预测未来。

输入：

- GPU 伪图像张量。

监督：

- trend。
- seasonality。
- forecastability。
- 8-cell regime。

可选：

- route soft label。

### Stage C: Few-shot Continuous Routing

目的：

> 在少量目标样本下，用视觉结构先验给出专家连续融合权重。

对照：

- fixed best expert。
- uniform ensemble。
- global weighted ensemble。
- statistical meta-feature router。
- TS encoder router。
- VisMoE-style hard router。
- visual continuous router。

关键指标：

- MSE/MAE/MASE/SMAPE。
- oracle gap。
- expert utilization entropy。
- routing-regime agreement。
- latency。

## 5. 当前最需要决策的三个问题

1. **MVP 的 few-shot 定义**
   建议：QuitoBench cell-level low-shot router adaptation。

2. **第一批候选专家**
   建议：先用 6-7 个候选做 profiling，再裁剪到 3-4 个专家。

3. **ViT 输入方案**
   建议：224 resize 和 `H x W` small patch embedding 同时跑，小尺寸方案不能只凭直觉确立。

## 6. 一句话版本

当前最稳的论文主张是：

> 在 TSF 分层的 QuitoBench 上，先通过专家 profiling 证明不同结构 cell 存在模型偏好，再用经过 TSF 语义微调的视觉编码器从 GPU 伪图像张量中提取结构先验，在 few-shot 条件下进行连续专家路由，并系统分析哪些伪图像和路由模块适合哪些时序结构。
