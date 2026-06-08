# Artifact Inventory

本文用于给当前工作区的主要文档、代码和输出产物打标签。目的不是删除文件，而是让后续开发知道哪些可以继承，哪些只作为审计证据保留，哪些只是 smoke/scratch。

状态标签：

| 标签 | 含义 |
| --- | --- |
| `canonical` | 后续主线可以继续复用 |
| `reference` | 作为事实证据、报告或解释材料保留 |
| `smoke` | 只证明代码链路跑通，不作为实验结论 |
| `deprecated` | 不建议继续沿用，仅保留历史证据 |
| `scratch` | 临时诊断或可重算中间产物 |

## 1. 文档资产

| 路径 | 状态 | 用途 |
| --- | --- | --- |
| `docs/STAGE_RESET_SUMMARY.md` | `canonical` | 当前正本清源总结，后续路线收敛依据 |
| `docs/CANONICAL_EXPERT_CACHE_REBUILD_PLAN.md` | `canonical` | 三专家同 registry clean cache rebuild 执行计划 |
| `docs/MATRIX50K_ERROR_PATHOLOGY_AUDIT.md` | `canonical` | clean matrix raw/normalized 排名分歧、missing/std/outlier/subset/cell 拆解 |
| `docs/WORKSPACE_CLEANUP_PLAN.md` | `canonical` | 工作区清理和归档执行计划 |
| `docs/WORKSPACE_MOVE_MANIFEST.csv` | `canonical` | 第一批建议移动/保留路径清单 |
| `docs/TOOL_ENTRYPOINTS.md` | `canonical` | `tools/` 入口分层、调用顺序和合并建议 |
| `quito/QUITO_CODEBASE_READING.md` | `canonical` | Quito 代码库复用指南 |
| `Doc/视觉伪图像路由双路线实施计划.md` | `reference` | 双路线方案历史设计 |
| `Doc/视觉路由实验路线对比_保守验证与推荐主线.md` | `reference` | 路线 1/2 关系与研究目标定义 |
| `Doc/视觉伪图像路由项目交接.md` | `reference` | Stage 1.2 前后的交接状态 |
| `experiment_logs/实验日志总览.md` | `reference` | stage 级索引和可信度粗判 |
| `experiment_logs/*.md` | `reference` | 具体实验日志；失败分支也保留作为决策证据 |

建议：不要删除 `experiment_logs/`。后续如果移动 outputs，应先确认日志引用路径是否需要补充说明。

## 2. 数据审计产物

`outputs/data_audit/` 整体建议保留。

| 路径 | 状态 | 用途 |
| --- | --- | --- |
| `quitobench_sufficiency_report.md` / `quitobench_window_counts.csv` | `reference` | 数据规模和窗口充分性 |
| `quitobench_tsf_label_source_report.md` | `canonical` | 官方 cluster 来源审计 |
| `quitobench_tsf_cells_final.csv` | `canonical` | item 级官方 cluster/final cell 表 |
| `quitobench_official_cluster_codebook.csv` | `canonical` | 官方 cluster code -> TSF cell 映射 |
| `quitobench_official_cluster_codebook_report.md` | `canonical` | codebook 反推证据 |
| `quitobench_channel_quality_stl_full.csv` | `reference` | 6,450 条通道级 full-length STL |
| `quitobench_item_quality_stl_channel_mean.csv` | `reference` | item 级 channel mean STL |
| `quitobench_official_codebook_channel_stl_validation*` | `reference` | Stage 0.7 codebook 验证 |
| `quitobench_official_cluster_semantics*` | `reference` | cluster 经验语义解释，不能替代 codebook |
| `quitobench_item_quality*.csv` / `quitobench_stl_quality_report.md` | `reference` | item 级 STL/proxy 历史审计 |

不建议删除这些文件。它们是后续路线 2 分层、解释和审稿答辩的主要证据。

## 3. 核心工具脚本

| 路径 | 状态 | 备注 |
| --- | --- | --- |
| `tools/quitobench_window_registry.py` | `canonical` | 生成 `physical_window_id/base_registry_id/sample_set_id` |
| `tools/quitobench_registry_subset.py` | `canonical` | 将固定抽样物化为一份新的 `window_registry`，用于三专家共用同一批 windows |
| `tools/quitobench_registry_audit.py` | `canonical` | 只读 registry 的 split/subset/cell/item/channel 分布审计 |
| `tools/quitobench_sample_channel_light_proxy.py` | `canonical` | 离线 proxy 与 torch online kernel |
| `tools/quitobench_imageization_protocol.py` | `canonical` | 三视图 imageization |
| `tools/quitobench_lightweight_expert_cache.py` | `canonical` | SNaive/轻量专家 cache schema |
| `tools/quitobench_framework_expert_cache.py` | `canonical` | DLinear/PatchTST/TSMixer wrapper；后续主线只建议用 DLinear/PatchTST |
| `tools/quitobench_expert_cache_audit.py` | `canonical` | 只读 expert cache，检查 key/horizon/manifest/common windows |
| `tools/quitobench_oracle_target_audit.py` | `canonical` | common-window best fixed / true uniform / oracle target audit |
| `tools/quitobench_normalized_oracle_audit.py` | `canonical` | 不改 cache，重建 Quito train-segment scaler 后计算 normalized-scale oracle audit |
| `tools/quitobench_expert_cache_comparison.py` | `canonical-needs-fix` | 可复用，但 `uniform_mse_proxy` 不是真 ensemble MSE |
| `tools/quitobench_visual_encoder_adapter_smoke.py` | `canonical` | visual embedding cache smoke 入口 |
| `tools/quitobench_quito_native_utils.py` | `canonical` | sample-weighted metric helper 等 |
| `tools/quitobench_common.py` | `canonical` | prediction columns、manifest JSON、列/key 校验、common expert-window 等低风险公共 helper |
| `tools/workspace_move_manifest_check.py` | `canonical` | 清理移动前的只读 manifest 检查 |
| `tools/quitobench_proxy_imageization_latency.py` | `reference` | latency sweep 工具；结果需干净环境重测 |
| `tools/quitobench_quito_native_sanity.py` | `reference` | 官方 dense/native sanity；交互式慎用 |
| `tools/quitobench_*cluster*.py` / `*stl*.py` | `reference` | Stage 0 审计工具，后续按需复跑 |
| `tools/quitobench_dlinear_expert_cache.py` | `deprecated` | 已被通用 `framework_expert_cache.py` 替代 |
| `tools/quitobench_expert_prediction_diagnostics.py` | `reference` | 诊断工具，不作为主流程入口 |

后续代码整理建议：先新增公共 helper，再逐步让主入口脚本复用，不要一次性重构所有 tools。

## 4. Window Registry

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506` | `canonical-candidate` | 后续推荐主线：96/48/S all-channel sparse |
| `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1` | `canonical` | 固定 50k sampled registry，三专家 clean matrix 共用同一 `window_index.csv` |
| `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/audit` | `canonical` | canonical candidate 的分布审计输出 |
| `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/audit` | `canonical` | fixed 50k registry 分布审计 |
| `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d` | `reference` | 96/48/S ind_1 sparse，对齐检查可用 |
| `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e` | `reference` | Stage 1.1/1.2 working registry |
| `outputs/vision_ts_routing/window_registry/qb_h576_p288_quito_overlap_d8cfe7ee_stride288_d9655deb` | `reference` | 576/288/S sparse sanity |
| `outputs/vision_ts_routing/_scratch/smoke/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1` | `smoke` | 测试输出，已从 `window_registry/` 迁出 |
| `outputs/vision_ts_routing/_scratch/smoke/window_registry/968e482b1cb0` | `smoke` | 少 item dense-ish smoke，已从 `window_registry/` 迁出 |
| `outputs/vision_ts_routing/_deprecated/window_registry/cfcd86e70e73` | `deprecated` | 旧 strict/coarse registry，不对齐 Quito overlap，已从 `window_registry/` 迁出 |

暂不移动目录。后续若归档，优先把 `cfcd86e70e73` 移到 `_deprecated/window_registry/`，但需先更新引用说明。

## 5. Proxy / Image / Visual Embedding

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| `outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e` | `canonical` | Stage 1.1 full proxy cache |
| `outputs/vision_ts_routing/_scratch/smoke/proxy_features/*__smoke*` | `smoke` | 测试用，已从 `proxy_features/` 迁出 |
| `outputs/vision_ts_routing/_scratch/smoke/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1` | `smoke-reference` | 三视图协议 smoke，设计可继承但不是正式 embedding set |
| `outputs/vision_ts_routing/_scratch/smoke/visual_embeddings/*/visual_embedding_cache_smoke_v1` | `smoke-reference` | 验证 `physical_window_id` 对齐 |
| `outputs/vision_ts_routing/_scratch/gpu_sanity/visual_embeddings_gpu_sanity/*` | `scratch` | GPU/IO 占用条件下的 sanity，不作性能结论 |
| `outputs/vision_ts_routing/latency` | `reference` | latency 工具输出；严肃结论需干净重测 |

下一阶段如果采用 `96/48/S all-channel` 主线，需要重新生成与该 registry 对齐的 image/embedding cache。

## 6. Expert Predictions

### 6.1 建议保留为主线候选

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/seasonal_naive_period6__official_align_h96_p48_allch_stride288_50k` | `canonical-candidate` | SNaive period=6 all-channel sparse |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/dlinear__official_align_h96_p48_allch_stride288_50k_e5_std` | `canonical-candidate` | DLinear + Quito scaler all-channel sparse |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/patchtst__official_align_h96_p48_allch_stride288_50k_e5_std` | `canonical-candidate` | PatchTST + Quito scaler all-channel sparse |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/seasonal_naive_period6__matrix50k_v1` | `canonical` | fixed registry 上的 SNaive period=6 cache |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/dlinear__matrix50k_v1_e20_std` | `canonical` | fixed registry 上的 DLinear e20 + Quito scaler cache |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/patchtst__matrix50k_v1_e20_std` | `canonical` | fixed registry 上的 PatchTST e20 + Quito scaler cache |
| `expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_6112373d/*official_align*h96_p48_ind1*` | `reference` | ind_1 对齐 sanity |

当前三份 all-channel 50k cache 的只读审计输出：

```text
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/official_align_h96_p48_allch_stride288_50k/
```

状态：`reference`。

重要结论：三者单 cache 内部 key/horizon 基本正常，但共同窗口只有 23,456 个，不能直接当作同一个 50k expert matrix。

这些仍是 sparse sanity，不是官方 full dense 复现。后续可以作为 canonical flow 的种子，但建议重建一次并补齐 true ensemble 指标。

### 6.2 只作 reference / diagnostics

| 路径模式 | 状态 | 原因 |
| --- | --- | --- |
| `expert_predictions/qb_h192_p96.../lightweight_v1__seasonal_naive_full` | `reference` | full seasonal baseline，可验证 schema 和低成本 baseline |
| `expert_predictions/qb_h192_p96.../lightweight_v1__stratified_50k` | `reference` | 轻量专家互补性分析 |
| `expert_predictions/qb_h192_p96.../dlinear_v1__stage14f*quito_scaler*` | `reference` | Quito scaler 对齐诊断 |
| `expert_predictions/qb_h192_p96.../patchtst_v1__stage14f*quito_scaler*` | `reference` | PatchTST 尺度诊断 |
| `expert_predictions/qb_h576_p288.../*stage14g_b*` | `reference` | 576/288 sparse sanity，暂非主线 |

### 6.3 建议降级或归档

| 路径模式 | 状态 | 原因 |
| --- | --- | --- |
| `expert_predictions/qb_h192_p96.../*stage14e*scaler*` | `deprecated` | wrapper-level global scaler，不再作为主口径 |
| `expert_predictions/qb_h192_p96.../*stratified_20k*` | `deprecated` | Stage 1.4d 稳定性诊断，口径已被后续覆盖 |
| `expert_predictions/qb_h192_p96.../*stratified_50k_cuda_e1/e5` | `deprecated` | 早期预算校准，raw/标准化口径混乱 |
| `expert_predictions/*/*_smoke*` | `smoke` | 跑通验证，不作结论 |
| `expert_predictions/qb_h96_p48.../*stage14g_b*h96_p48_stride288_50k_e5` | `deprecated` | raw sparse + period/normalize 口径未对齐 |

## 7. Expert Comparisons

| 路径模式 | 状态 | 说明 |
| --- | --- | --- |
| `oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/common_23456_official_align` | `canonical` | 当前三专家 23,456 个共同窗口上的 raw-scale oracle target audit |
| `expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1` | `canonical` | clean 三专家 cache 审计；common prediction/error windows 均为 50,000 |
| `oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1` | `canonical` | clean matrix raw-scale oracle audit |
| `oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1_normalized` | `canonical-needs-debug` | clean matrix normalized-scale oracle audit；与 raw-scale 排名分歧，需要排查后再 gate |
| `oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/pathology_matrix50k_v1` | `canonical` | per-window/expert raw+normalized error、std、subset/cell/outlier 分解 |
| `oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/data_missing_summary.json` | `canonical` | hour/min 原始 parquet NaN/inf 审计 |
| `expert_comparisons/qb_h192_p96.../stage14f_h192_p96_20k_e20_lr1e4_quito_scaler` | `reference` | Quito scaler 诊断 |
| `expert_comparisons/qb_h192_p96.../stage14e*` | `deprecated` | wrapper scaler 过渡口径 |
| `expert_comparisons/qb_h192_p96.../stage1_4d*` | `deprecated` | 训练稳定性诊断，不作主结论 |
| `expert_comparisons/qb_h192_p96.../budget_calibration*` | `deprecated` | 预算校准，已被 scaler 对齐结论覆盖 |

注意：当前 comparison 中 `uniform_mse_proxy` 是专家 MSE 平均，不是真正把预测平均后再算 MSE。后续 canonical comparison 应修正。

## 8. 建议的清理顺序

第一阶段，不移动文件：

1. 使用本文档标记可信度。
2. 在新实验中只读取 `canonical` 和 `canonical-candidate`。
3. 文档中引用 deprecated 结果时明确“只作历史诊断”。

第二阶段，等 canonical flow 重跑通过后：

1. 创建 `outputs/vision_ts_routing/_deprecated/`。
2. 移动旧 strict registry、Stage 1.4d/e raw/global scaler、大量 smoke cache。
3. 在本文档追加迁移记录。

第三阶段，代码整理：

1. 新增 `tools/quitobench_common.py`。
2. 合并 registry validation、history/target extraction、sampling、manifest helper。
3. 保持现有 CLI 参数兼容，减少旧日志命令失效。

## 9. 迁移记录

2026-06-08 已执行第一批可逆迁移，依据 `docs/WORKSPACE_MOVE_MANIFEST.csv`：

| 旧路径 | 新路径 |
| --- | --- |
| `outputs/vision_ts_routing/window_registry/cfcd86e70e73` | `outputs/vision_ts_routing/_deprecated/window_registry/cfcd86e70e73` |
| `outputs/vision_ts_routing/window_registry/968e482b1cb0` | `outputs/vision_ts_routing/_scratch/smoke/window_registry/968e482b1cb0` |
| `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1` | `outputs/vision_ts_routing/_scratch/smoke/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1` |
| `outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_rows_2000` | `outputs/vision_ts_routing/_scratch/smoke/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_rows_2000` |
| `outputs/vision_ts_routing/visual_embeddings_gpu_sanity` | `outputs/vision_ts_routing/_scratch/gpu_sanity/visual_embeddings_gpu_sanity` |

2026-06-08 已执行第二批可逆迁移，完整路径见 `docs/WORKSPACE_MOVE_MANIFEST.csv`：

| 类别 | 数量 | 新位置 |
| --- | ---: | --- |
| deprecated expert predictions | 17 | `outputs/vision_ts_routing/_deprecated/expert_predictions/` |
| smoke / 1k diagnostic expert predictions | 17 | `outputs/vision_ts_routing/_scratch/smoke/expert_predictions/` |
| deprecated expert comparisons | 8 | `outputs/vision_ts_routing/_deprecated/expert_comparisons/` |
| smoke expert comparisons | 2 | `outputs/vision_ts_routing/_scratch/smoke/expert_comparisons/` |

第二批保留未移动：

- `stage14f_h192_p96_20k_e20_lr1e4_quito_scaler` 相关 expert prediction/comparison，作为 Quito scaler reference。
- `lightweight_v1__seasonal_naive_full` 和 `lightweight_v1__stratified_50k`，作为轻量 schema/reference。
- `qb_h96_p48.../official_align*` 三专家 cache，作为当前 canonical-candidate。
- `qb_h576_p288.../*stage14g_b*`，作为 576/288 sparse sanity reference。

2026-06-08 已执行第三批可逆迁移，完整路径见 `docs/WORKSPACE_MOVE_MANIFEST.csv`：

| 类别 | 数量 | 新位置 |
| --- | ---: | --- |
| image tensor smoke-reference | 1 | `outputs/vision_ts_routing/_scratch/smoke/image_tensors/` |
| visual embedding smoke-reference | 1 | `outputs/vision_ts_routing/_scratch/smoke/visual_embeddings/` |
| Quito native smoke | 1 | `outputs/vision_ts_routing/_scratch/smoke/quito_native_sanity/` |

第三批保留未移动：

- `outputs/vision_ts_routing/latency`，作为 latency 工具输出 reference。
- `outputs/vision_ts_routing/run_logs`，先保留，后续需核对实验日志引用后再决定是否归档。
