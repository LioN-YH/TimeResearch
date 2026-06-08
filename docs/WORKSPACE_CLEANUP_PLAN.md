# Workspace Cleanup Plan

本文把 `docs/ARTIFACT_INVENTORY.md` 的清理规则落成可执行计划。目标是让工作区变干净，但不删除历史证据，不破坏仍可复查的实验链路。

## 1. 原则

- 不删除 outputs、日志或历史脚本。
- 不移动 `canonical` / `canonical-candidate` / 关键 `reference`。
- 先隐藏 `deprecated`、`smoke`、`scratch`，而不是清空。
- 移动前保留迁移清单，移动后在 `ARTIFACT_INVENTORY.md` 追加迁移记录。
- 旧日志中的路径不批量改写；迁移清单负责说明旧路径到新路径的映射。

## 2. 当前阶段

当前处于第一阶段和第二阶段之间：

1. 第一阶段：资产标记已完成，见 `docs/ARTIFACT_INVENTORY.md`。
2. 第二阶段：可以开始准备 `_deprecated` / `_scratch` 归档目录，但不建议一次性移动所有产物。
3. 第三阶段：代码合并暂缓，先等主线工具稳定后再抽 `tools/quitobench_common.py`。

## 3. 推荐目录结构

建议在 `outputs/vision_ts_routing/` 下建立：

```text
_deprecated/
  window_registry/
  expert_predictions/
  expert_comparisons/
_scratch/
  smoke/
  gpu_sanity/
```

含义：

- `_deprecated/`：不建议后续读取，只保留历史证据。
- `_scratch/`：临时 smoke、GPU/IO sanity、可重算中间产物。
- `reference` 默认保留在原位，除非它明显污染主线入口。

## 4. 第一批已执行移动

第一批只移动最确定不应作为主线输入的目录，已于 2026-06-08 执行：

| 旧路径 | 新路径 | 原因 |
| --- | --- | --- |
| `outputs/vision_ts_routing/window_registry/cfcd86e70e73` | `outputs/vision_ts_routing/_deprecated/window_registry/cfcd86e70e73` | 旧 strict/coarse registry，不对齐 Quito overlap |
| `outputs/vision_ts_routing/window_registry/968e482b1cb0` | `outputs/vision_ts_routing/_scratch/smoke/window_registry/968e482b1cb0` | 少 item dense-ish smoke |
| `outputs/vision_ts_routing/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1` | `outputs/vision_ts_routing/_scratch/smoke/window_registry/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_items_1` | registry smoke |
| `outputs/vision_ts_routing/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_rows_2000` | `outputs/vision_ts_routing/_scratch/smoke/proxy_features/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__smoke_max_rows_2000` | proxy smoke |
| `outputs/vision_ts_routing/visual_embeddings_gpu_sanity` | `outputs/vision_ts_routing/_scratch/gpu_sanity/visual_embeddings_gpu_sanity` | GPU/IO 条件下 sanity，不作性能结论 |

## 5. 暂不移动

以下目录先保持原位：

- `outputs/data_audit/`：路线 2 和论文解释证据。
- `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506`：当前主线候选。
- `outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506`：当前三专家候选 cache。
- `outputs/vision_ts_routing/oracle_audit/`：当前 common-window target audit。
- `experiment_logs/`：保留历史决策证据，不移动。

## 6. 第二批已执行移动

第二批已于 2026-06-08 执行，完整路径见 `docs/WORKSPACE_MOVE_MANIFEST.csv`。本批移动对象：

- deprecated expert predictions: 17 个目录。
- smoke / 1k diagnostic expert predictions: 17 个目录。
- deprecated expert comparisons: 8 个目录。
- smoke expert comparisons: 2 个目录。

本批明确保留：

- Stage 1.4f Quito scaler reference。
- 当前 h96/p48 official alignment 三专家 cache。
- h576/p288 sparse sanity reference。
- `lightweight_v1__seasonal_naive_full` 和 `lightweight_v1__stratified_50k`。

## 7. 第三批已执行移动

第三批已于 2026-06-08 执行，完整路径见 `docs/WORKSPACE_MOVE_MANIFEST.csv`。本批移动对象：

- Stage 1.2 image tensor smoke-reference: 1 个目录。
- Stage 1.3 visual embedding smoke-reference: 1 个目录。
- Quito native SNaive smoke: 1 个目录。

本批明确保留：

- `outputs/vision_ts_routing/latency`，作为 latency reference。
- `outputs/vision_ts_routing/run_logs`，先保留等待日志引用核对。

## 8. 代码整理建议

暂不大规模重构 `tools/`。建议分三步：

1. 只把入口身份写清楚：canonical / compatibility / reference / deprecated。
2. 等下一轮主线工具稳定后新增 `tools/quitobench_common.py`。
3. 逐步合并重复函数：
   - registry validation
   - prediction column discovery
   - cache manifest loading
   - history/target extraction
   - stratified sampling
   - output manifest writing

当前不建议删除 `tools/quitobench_dlinear_expert_cache.py`，因为它是兼容入口；但文档中应继续标记为 deprecated wrapper。

## 9. 执行方式

后续新增移动项前先检查：

```bash
python tools/workspace_move_manifest_check.py
```

如果没有该脚本，则人工按 `docs/WORKSPACE_MOVE_MANIFEST.csv` 检查：

- old_path 是否存在；
- new_path 是否不存在；
- 状态是否为 `ready`；
- 是否属于第一批移动。

移动后追加记录到 `docs/ARTIFACT_INVENTORY.md` 的清理章节。第一批迁移已经记录在 `docs/ARTIFACT_INVENTORY.md` 的“迁移记录”。
