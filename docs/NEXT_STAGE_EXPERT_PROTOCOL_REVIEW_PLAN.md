# 下一阶段专家协议与 Router 数据构造评审计划

> **给后续执行者的要求：** 本文是阶段评审计划，不是继续堆实验的清单。执行时应先完成审计和小规模诊断，再决定是否重建统一 expert matrix。任何进入 visual router / gate 训练的动作，都必须先满足本文的评审门槛。

**目标：** 重新评审 SNaive / DLinear / PatchTST 是否适合作为下一阶段视觉 router 的专家集合，并确定一个不会退化为 item/cell lookup 的统一数据构造协议。

**核心判断：** 最近 dense/sparse 结果显示，当前混合训练/采样/聚合协议下 PatchTST/DLinear 表现不稳定，不能继续默认三专家组合已经成立。但 per-cell smoke 也显示 PatchTST 在 5/8 个代表 item 上最好，因此不能简单放弃神经专家。下一阶段要判断的是专家互补性是否能在统一、平衡、out-of-item 协议下稳定出现。

**技术栈：** Quito `TimeSeriesDataset` dense loader、现有 registry/cache 工具、normalized-scale MAE/MSE、item/cell macro 聚合、Python/pytest。

---

## 1. 当前需要暂停的动作

暂停：

- 继续训练 visual embedding / gate。
- 基于 `matrix50k_v1` 直接生成正式 router 训练集。
- 继续调 PatchTST/DLinear epoch/lr 来解释倒挂。
- 训练 per-cell router 或把 cell label 作为模型输入。
- 每个 item 单独训练专家后直接收集 router 标签。

原因：

- `matrix50k_v1` 已经可作为 pipeline smoke，但不适合作为复现结论或正式 router 主训练集。
- 8-cell mixed dense smoke 中 SNaive 仍然 MAE 最低，说明混合平均会被少数 item/cell 拉偏。
- per-cell smoke 显示 PatchTST 在多数代表 item 上最好，说明“PatchTST 全局失效”不是正确结论。
- 如果每 item 单独训练专家，router 会学到 item/model 关系，而不是可迁移的 history-window 结构先验。

## 2. 本阶段要回答的问题

### Q1：专家是否有稳定互补性？

判断标准不是“PatchTST 是否总是最好”，而是：

- 至少两个专家在不同 item/cell/window 上有稳定 top1 区域；
- oracle 相比 best fixed 有非平凡收益；
- 该收益在 item-level holdout 上仍存在；
- router 输入不含 item id / cell label 时仍有可学习信号。

### Q2：混合训练失败来自哪里？

候选原因：

- mixed train pool 太小或代表性偏；
- sample-window weighted mean 被长序列或异常 item 主导；
- lowT/lowS 或 lowT/lowS/highF cell 中存在 SNaive-friendly item；
- PatchTST/DLinear 的训练协议还没有对齐官方 checkpoint/evaluate 方式；
- 当前统一训练集没有 item/cell balancing。

### Q3：最终 router 数据构造是否保持通用性？

必须满足：

- 专家训练是统一协议，不是每 item 一个专家；
- router 训练、验证、测试按 item 分组切分；
- cell label 只用于采样平衡和诊断报告，不作为 router 输入；
- visual imageization 只来自 history window，不看 target；
- oracle / top1 标签只来自 held-out target error。

## 3. 已确认事实

### 3.1 Quito 协议事实

- Quito 标准化是 train segment fit，粒度为 item/channel。
- valid/test history 可以 overlap 前一 split 尾部，target 不 overlap。
- 官方 evaluate 更接近 normalized-scale metric，不做 inverse transform 后再报 Table 24。
- 当前没有源码证据支持官方 per-cell standardization。

### 3.2 当前实验事实

- 旧三份 50k cache 不是同一窗口集合，已不再作为 expert matrix 使用。
- `matrix50k_v1` 三专家窗口已对齐，但 mixed sparse 结果仍与论文排序不一致。
- 2-item dense smoke 恢复 `PatchTST > DLinear > SNaive`。
- 8-cell mixed dense smoke 未恢复排序，SNaive p=6 MAE 最低。
- per-cell single-item dense smoke 中 PatchTST 在 5/8 个代表 item 上最好，SNaive 在 3/8 个 lowT 相关代表 item 上最好。
- `tools/quitobench_dense_smoke.py` 曾存在 `ids: []` 误加载全量 subset 的 bug，已修复并测试。

## 4. 推荐评审流程

### Task A：扩展 per-cell 代表 item 诊断

**目的：** 判断 3 个 SNaive-best cell 是否是单 item 偶然性。

**输入：**

```text
outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv
```

**选择规则：**

- `official_tsf_cell == paper_like_tsf_cell`
- 每 cell 选择 2-3 个 high-margin item
- hour/min 尽量都覆盖
- 不使用 target error 选择 item

**输出：**

```text
outputs/vision_ts_routing/quito_dense_smoke/per_cell_stage07_multiitem/
```

**验收：**

- 每个 cell 至少 2 个 item；
- 每个 item 输出 SNaive/DLinear/PatchTST normalized MAE；
- 汇总 item-macro、cell-macro、winner count；
- 明确标记哪些 cell 是稳定 SNaive-friendly。

### Task B：统一 dense-inspired train pool 设计

**目的：** 构造一个统一训练池，不做 per-item 专家训练。

**设计原则：**

- train pool 按 cell/item/subset 平衡；
- 每个 item 取 dense rolling train windows 的可控子样本；
- valid/test 采用 item-level holdout 或至少 item-disjoint audit split；
- 不把 cell label 写入 router input，只保留在 manifest/audit 中。

**待产物：**

```text
docs/DENSE_INSPIRED_REGISTRY_DESIGN.md
```

内容必须包含：

- item 选择规则；
- dense window 降采样规则；
- split 规则；
- item/cell/subset quota；
- 专家训练是否能看到 test item；
- router train/valid/test 是否 item-disjoint。

### Task C：统一专家训练协议评审

**目的：** 决定 DLinear/PatchTST 是否继续作为主专家。

**必须比较三种协议：**

| protocol | 含义 | 是否可作为 router 主协议 |
| --- | --- | --- |
| per-item train/test | 每 item 单独训练专家 | 否，只能诊断 |
| mixed unbalanced train | 当前 mixed smoke 类协议 | 暂不建议 |
| unified balanced train | 统一训练池 + item/cell balanced | 候选主协议 |

**验收：**

- 若 unified balanced 下 PatchTST/DLinear 仍缺乏 top1 区域，应降级或替换；
- 若 oracle 相比 best fixed 收益很小，应暂停 router，先改 expert pool；
- 若收益只在 seen item 上存在，应重做 item-disjoint split。

### Task D：Router 数据构造门槛

进入 visual router 前必须满足：

- 专家 cache 来自同一 unified registry；
- prediction/error key 完全对齐；
- normalized MAE/MSE、raw MAE/MSE 都有；
- 报告 item-macro、cell-macro、sample-window weighted 三种聚合；
- 至少有一个 item-disjoint validation/test 报告；
- expert oracle 在 validation/test 上相对 best fixed 有稳定增益；
- cell label 不作为 router input。

## 5. 专家池调整原则

如果 DLinear/PatchTST 在统一平衡协议下仍不稳定，不应继续为了维护三专家设定而调参。

可选调整：

- 保留 SNaive p=6 作为 baseline；
- 增加多个 seasonal period 的 naive baseline；
- 增加 decomposition / smoothing / statistical experts；
- 将 DLinear/PatchTST 降级为候选专家，而不是主专家；
- 优先选择能在 item-disjoint split 上提供互补 top1 区域的专家。

视觉 router 的目标是学习“什么时候该信谁”，不是证明某个指定模型必须强。

## 6. 与通用视觉先验目标的关系

本阶段使用 cell/item 诊断不会改变最终目标。

限制：

- cell label 不进入视觉模型；
- item id 不进入视觉模型；
- per-cell/per-item 只用于审计、采样平衡和失败定位；
- 最终训练仍应使用统一 registry；
- 泛化评估必须包含未见 item。

保留目标：

> 从 sample-channel history window 的图像化表示中学习可迁移的时序结构先验，用该先验在多个异构专家之间做连续自适应融合。

## 7. 建议下一步

1. 先完成每 cell 2-3 个 exact-match item 的 dense smoke。
2. 汇总 item-macro/cell-macro 结果，判断 SNaive-best cell 是否稳定。
3. 写 `docs/DENSE_INSPIRED_REGISTRY_DESIGN.md`，明确统一平衡 registry 设计。
4. 决定专家池是否仍采用 SNaive / DLinear / PatchTST，还是替换或扩展。
5. 只有通过 Router 数据构造门槛后，才恢复 visual embedding / gate 实验。
