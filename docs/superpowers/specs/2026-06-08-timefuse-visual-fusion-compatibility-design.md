# TimeFuse 视觉融合兼容性审计设计

## 1. 背景

当前 QuitoBench 主线已经建立了 registry、专家 cache、oracle audit 和 visual embedding smoke，但实验重心被 raw/normalized 尺度分歧、DLinear/PatchTST outlier 和 benchmark 口径诊断牵制。用户当前优先目标是验证：

> 视觉 embedding 对时序专家融合是否有增益。

因此下一步不应继续把主要精力放在 QuitoBench 复现细节上，而应先找一个更接近 prediction-level fusion 的平台验证核心命题。TimeFuse 的公开实现正好以 sample-level adaptive model fusion 为目标，核心数据协议包含：

- `x_meta`: 输入窗口的统计/结构 meta feature。
- `y_model_preds`: 多个 base model 对同一 target 的预测，形状为 `[N, K, pred_len, channels]`。
- `y_true`: target 真值，形状为 `[N, pred_len, channels]`。

这与当前项目需要的 expert matrix、visual embedding 和 gate/fusion baseline 高度一致。

## 2. 目标

第一阶段只做 TimeFuse 兼容性审计和最小迁移设计，不直接改 TimeFuse 训练主流程。

必须回答：

1. TimeFuse 是否能稳定提供 `x_meta / y_model_preds / y_true` 三元组。
2. 这些数组是否能映射到当前项目的 `sample_id / expert_id / yhat_* / target_* / error` 评估口径。
3. 当前三视图视觉协议是否能在 TimeFuse 样本粒度上生成 `z_*` embedding，并与 `x_meta` 一一对齐。
4. 视觉增强 fusor 的最小对比是否能定义为严格的 ablation，而不是代码路径差异。

暂不做：

- 不训练 TimeFuse base models。
- 不修改 TimeFuse 上游模型代码。
- 不把 QuitoBench 的 `physical_window_id` schema 强行搬进 TimeFuse。
- 不追求 TSF cell 覆盖作为第一阶段成功条件。
- 不下载或运行超出 TimeFuse README 所需的数据包和 checkpoint 之外的大规模资源。

## 3. 推荐路线

采用“双层验证”路线：

1. TimeFuse 作为主验证平台，验证视觉 embedding 是否提升 sample-level model fusion。
2. QuitoBench 保留为结构解释平台，后续只用于分层解释或外部验证，不再作为第一阶段 fusion 增益验证的阻塞项。

这样可以把论文叙事拆开：

- TimeFuse 回答“视觉 embedding 对专家融合是否有增益”。
- QuitoBench 回答“这种增益是否和 TSF cell、结构类型、通道异质性相关”。

## 4. TimeFuse 当前接口判断

本地已克隆仓库：

```text
TimeFuse/
```

当前主分支 HEAD：

```text
978e6c6b9e4f246632c269aa0f9beeb099eabcfc
```

关键文件：

- `TimeFuse/timefuse.py`: `Dataset_Meta` 加载 meta arrays，`ModelFusor` 由 meta features 输出模型权重。
- `TimeFuse/meta_feature.py`: 统计、ACF、ADF、频域、协方差等 hand-crafted meta feature。
- `TimeFuse/exp/exp_fuse_forecasting.py`: base model 训练、测试、提取 test meta feature。
- `TimeFuse/run_timefuse_exp.ipynb`: notebook 形式的完整实验流程。
- `TimeFuse/run_config.json`: 默认数据集为 `ETTh1`、`ETTh2`，默认模型为 `DLinear`、`PatchTST`、`TimesNet`、`PAttn`、`TimeMixer`、`TimeXer`。

当前 clone 不包含：

- `dataset/`
- `meta_data/`
- `checkpoints/`

这些需要按 README 下载 Google Drive 数据包后放入 TimeFuse 根目录。

## 5. 兼容性审计设计

### 5.1 Artifact 审计

检查 TimeFuse 数据包落盘后是否存在：

```text
TimeFuse/dataset/
TimeFuse/meta_data/
TimeFuse/checkpoints/
```

对默认 `ETTh1/ETTh2`、`forecast_setting=[96,48,96]`，至少检查：

```text
meta_data/ETTh1_val/x_meta_96.h5
meta_data/ETTh1_val/y_pred_96_48_96.h5
meta_data/ETTh1_val/y_true_96_48_96.h5
meta_data/ETTh1_test/x_meta_96.h5
meta_data/ETTh1_test/y_pred_96_48_96.h5
meta_data/ETTh1_test/y_true_96_48_96.h5
meta_data/ETTh2_val/...
meta_data/ETTh2_test/...
```

通过标准：

- `x_meta` 行数与 `y_pred/y_true` 样本数一致，或只存在 TimeFuse 已知的 `x_meta` 前缀截断情况。
- `y_pred` 形状为 `[N, K, pred_len, C]`。
- `y_true` 形状为 `[N, pred_len, C]`。
- 所有数组 finite，无 NaN/inf。
- `K` 与 `run_config.json` 中模型数量一致。

### 5.2 Expert Matrix 映射

为 TimeFuse 样本定义本地兼容 ID：

```text
timefuse_sample_id = <dataset>__<split>__sl<seq_len>_ll<label_len>_pl<pred_len>__row<row_idx>
```

该 ID 只用于本项目适配层，不修改 TimeFuse 原始代码。

从 `y_model_preds` 派生 long-format expert prediction table：

```text
timefuse_sample_id
dataset
split
seq_len
label_len
pred_len
channel_index
expert_id
yhat_0 ... yhat_{pred_len-1}
target_0 ... target_{pred_len-1}
```

同时派生 error table：

```text
timefuse_sample_id
dataset
split
channel_index
expert_id
mse
mae
```

如果第一版保留 multivariate channel 共同融合，则 `channel_index="all"`，error 在 `[pred_len, C]` 上整体计算。若后续需要与 QuitoBench sample-channel 口径统一，再展开为单 channel。

### 5.3 Visual Embedding 对齐

TimeFuse 第一版视觉输入粒度使用 `batch_x` 的完整 multivariate history window，形状为 `[seq_len, C]`。

迁移当前视觉部分时，先不使用 QuitoBench 的 `physical_window_id` 和原有 registry。应新增 TimeFuse 专用的轻量 imageization adapter：

```text
input:  X_in [seq_len, C]
output: view_tensor [V, H, W]
```

第一版可以保留三视图思想：

- line raster: 对每个 channel 或 channel mean 绘制归一化轨迹。
- period fold: 对主周期或固定周期进行折叠视图。
- fft power: 频域功率视图。

输出 embedding table：

```text
timefuse_sample_id
encoder_id
z_0 ... z_{D-1}
```

通过标准：

- embedding 行数与 `x_meta` 样本数一致。
- `timefuse_sample_id` 唯一。
- embedding finite。
- 同一 split 内可与 `y_pred/y_true` 完整 join。

### 5.4 Fusion Ablation

为了证明视觉 embedding 是否有增益，第一版只允许修改 fusor 输入，不修改 base model 预测和 target。

至少比较：

| baseline | fusor input |
| --- | --- |
| best single | 无训练，分别在每个 evaluation split 上报告最优单模型 reference，不作为可部署模型选择流程 |
| uniform | 无训练，平均所有 base model predictions |
| TimeFuse meta-only | 原始 `x_meta` |
| visual-only | `z_*` |
| meta+visual | `concat(x_meta, z_*)` |

训练集：

- 使用 TimeFuse notebook 当前口径：`*_val` 作为 meta-train，`*_test` 作为 meta-test。
- 如果发现 val/test 命名不适合论文式报告，应在日志中明确改名为 `meta_train_split` 和 `meta_eval_split`，避免混淆。

指标：

- MSE
- MAE
- best single gap
- uniform gap
- oracle top1 gap
- per-dataset result
- per-model weight utilization entropy

通过标准：

- `meta+visual` 在至少一个默认数据集上优于 `meta-only`，且没有在另一默认数据集上明显退化。
- 若没有提升，也必须能输出可信负结果：visual-only、meta-only、meta+visual 的训练路径完全一致，只是输入特征不同。

## 6. 实现边界

本支线代码必须与 QuitoBench 主线隔离，便于后续整体剥离。新增代码不放入根目录 `tools/` 或根目录 `tests/`，统一放入：

```text
timefuse_visual_fusion/
```

推荐结构：

```text
timefuse_visual_fusion/
  README.md
  src/timefuse_visual_fusion/
    __init__.py
    common.py
    artifact_audit.py
    matrix_export.py
    visual_embedding_smoke.py
    fusion_ablation.py
  tests/
    test_common.py
    test_artifact_audit.py
    test_matrix_export.py
    test_visual_embedding_smoke.py
    test_fusion_ablation.py
  outputs/
```

如果需要复用既有 QuitoBench helper，只复制必要的轻量逻辑到 `timefuse_visual_fusion/src/timefuse_visual_fusion/common.py`，不要 import `tools.quitobench_common`。第一版只需要复制/重写 JSON manifest 读写、required columns、unique key 这类小 helper。

输出目录建议放在支线内部：

```text
timefuse_visual_fusion/outputs/
```

不要把 TimeFuse 输出混入现有 `outputs/vision_ts_routing/` 主线，也不要把支线测试混入根目录 `tests/`。这样后续若确认 TimeFuse 成为主验证平台，可以把 `timefuse_visual_fusion/` 单独初始化为仓库或迁出当前工作区。

## 7. 风险与处理

### 数据包不可下载

如果 Google Drive 数据包无法下载，先不训练 base models。改为用已有 QuitoBench clean matrix 验证 TimeFuse fusor 形式，作为临时 fallback。

### 依赖版本冲突

TimeFuse `requirements.txt` 指定 `torch==1.7.1`，与当前 Quito 环境的 torch 2.6 不一致。第一版不应污染 `quito` 环境，应使用独立环境或先只运行不依赖旧 torch 的数组审计脚本。

### 视觉协议不适配 multivariate 输入

如果三视图协议对 `[seq_len, C]` 的多变量窗口效果不稳定，第一版可先使用 channel mean 或目标 channel，再把 channel-aware imageization 作为后续消融。

### TSF cell 说服力下降

TimeFuse 默认数据集没有 QuitoBench 官方 8-cell 标签。第一阶段报告中不能声称覆盖 8-cell，只能声称完成 prediction-level fusion 验证。TSF cell 分析保留给 QuitoBench 外部验证阶段。

## 8. 第一阶段完成标准

第一阶段结束时应产出：

1. TimeFuse artifact audit 报告。
2. TimeFuse expert matrix 导出结果。
3. visual embedding smoke cache。
4. `meta-only / visual-only / meta+visual` fusion ablation 表。
5. 一份实验日志，明确是否继续把 TimeFuse 作为主验证平台。

只有当第 1-3 项通过后，才进入 fusion ablation。只有当 fusion ablation 的路径稳定后，才考虑扩大数据集或把 QuitoBench TSF cell 重新接入解释线。
