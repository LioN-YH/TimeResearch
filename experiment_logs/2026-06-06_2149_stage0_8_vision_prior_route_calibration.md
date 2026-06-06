# Stage 0.8：视觉先验路线校准与实验协议细化

## 1. 实验目的

在路线 1/路线 2 正式实现之前，根据用户明确的研究目标“视觉模块学习跨窗口、跨 horizon、跨数据源稳定的时序结构先验”，校准现有方案文档和交接文档。

本阶段只做文档路线校准，不实现 router，不运行长实验，不修改 Quito 官方代码。

## 2. 实验计划

1. 阅读 Stage 0.7 通道级 full-length STL 官方 codebook 验证日志。
2. 将 Stage 0.7 结论写入后续路线约束。
3. 更新双路线实施计划，补充全局视觉时序结构先验目标、连续异构专家融合架构、Stage 0.8 和 Stage 1.0-1.5 任务定义。
4. 更新路线对比文档，把路线定位从“ViT router”校准为“视觉时序结构 encoder + continuous gate + 异构专家融合”。
5. 更新交接文档，说明 Stage 0.7/0.8 已完成、下一步不直接实现 router，而是进入 Stage 1.0。
6. 更新实验日志总览。

## 3. 执行命令

读取文档和日志使用 `sed`；文档编辑首选 `apply_patch`，但当前沙箱 helper 报错：`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，因此改用外部沙箱授权后的受控 Python 文本替换。

## 4. 输入数据与配置

输入文档：

- `Doc/视觉伪图像路由双路线实施计划.md`
- `Doc/视觉路由实验路线对比_保守验证与推荐主线.md`
- `Doc/视觉伪图像路由项目交接.md`
- `experiment_logs/2026-06-06_1950_stage0_7_channel_stl_codebook_validation.md`

关键 Stage 0.7 结论：

- 通道级 full-length STL 覆盖 1,290 个 item、6,450 条通道序列。
- channel-mean + `tau=0.4` 与 Stage 0.6b 官方 codebook 的 item exact match 为 68.37%。
- T/S/F 逐维匹配分别为 81.32% / 69.30% / 100.00%。
- cluster 24 在本阶段 100% 匹配官方 `lowT_lowS_highF`。
- cluster 6、8、26 仍存在 trend/seasonality 本地口径差异。

## 5. 实验结果

已更新：

- `Doc/视觉伪图像路由双路线实施计划.md`
- `Doc/视觉路由实验路线对比_保守验证与推荐主线.md`
- `Doc/视觉伪图像路由项目交接.md`
- `experiment_logs/2026-06-06_2149_stage0_8_vision_prior_route_calibration.md`
- `experiment_logs/实验日志总览.md`

主要校准内容：

1. 明确主目标：视觉模块学习跨 `history_len`、`pred_len`、subset/data source 稳定的时序结构先验。
2. 明确 sample-channel history window 是视觉模块实际输入粒度。
3. 明确 official cluster 是 item 级、多变量、全长解释标签，不作为 sample-window router 的硬监督。
4. 明确 Stage 0.7 channel STL 是通道异质性分析和伪图像设计参考，不替代 Stage 0.6b 官方 codebook。
5. 明确最终模型是 `visual time-series encoder + continuous gate + heterogeneous forecasting experts`。
6. 明确第一版采用冻结专家预测缓存训练 gate，不做端到端异构专家联合训练。
7. 明确在线路径不使用 full STL，而使用轻量 proxy、视觉 embedding 和 top-k soft routing。
8. 新增 Stage 1.0-1.5 任务顺序。

## 6. 问题与观察

- 当前路线文档此前更偏向“ViT 作为 router”，容易把视觉模块目标收窄到单一窗口设置下的专家标签拟合；本阶段已改为“视觉结构先验 encoder 是主体，router/gate 是下游验证器”。
- Stage 0.7 支持官方 codebook 的 forecastability 维度和 cluster 24，但 trend/seasonality 仍有本地口径差异，因此后续不能把本地 STL cell 当作官方标签替代品。
- `apply_patch` 工具在当前沙箱环境失败，本阶段文档更新改用受控 Python 文本替换完成。

## 7. 结论

Stage 0.8 已完成。

后续路线应从 Stage 1.0 开始，先做窗口索引与配置注册表，而不是直接实现 router。router 的训练目标应来自异构专家在同一 sample-channel/horizon 下的预测误差和 soft oracle；TSF cell 主要用于分层报告、few-shot 结构实验和结果解释。

## 8. 下一步计划

1. 等待用户确认文档校准是否符合预期。
2. 下一阶段建议实现 Stage 1.0：窗口索引与配置注册表。
3. Stage 1.0 完成后再做 Stage 1.1：sample-channel light proxy 预计算。
4. 继续避免直接实现 router，直到窗口索引、proxy、伪图像协议和专家预测缓存接口稳定。
