# Tool Entrypoints

本文说明当前 `tools/` 下脚本的角色和后续使用建议。目标是避免继续从 Stage 1.4 之后的历史分支里随手挑脚本跑，造成口径再次发散。

## 1. 当前主线入口

这些脚本可以作为后续主线继续复用。

| 脚本 | 角色 | 何时运行 | 测试 |
| --- | --- | --- | --- |
| `tools/quitobench_window_registry.py` | 生成 sample-channel window registry | 需要新建 canonical/sparse registry 时 | `tests/test_quitobench_window_registry.py` |
| `tools/quitobench_registry_subset.py` | 物化固定 sampled registry | 需要让多个专家复用同一批 sampled windows 时 | `tests/test_quitobench_registry_subset.py` |
| `tools/quitobench_registry_audit.py` | 只读 registry 分布审计 | 新 registry 生成后立即运行 | `tests/test_quitobench_registry_audit.py` |
| `tools/quitobench_sample_channel_light_proxy.py` | history-only light proxy cache / torch kernel | 需要 proxy cache 或 online proxy smoke 时 | `tests/test_quitobench_sample_channel_light_proxy.py` |
| `tools/quitobench_imageization_protocol.py` | 三视图 imageization tensor | 需要重新生成 view tensor 时 | `tests/test_quitobench_imageization_protocol.py` |
| `tools/quitobench_lightweight_expert_cache.py` | SNaive/轻量统计专家 cache | 需要 SNaive 或低成本 baseline cache 时 | `tests/test_quitobench_lightweight_expert_cache.py` |
| `tools/quitobench_framework_expert_cache.py` | DLinear/PatchTST/TSMixer 训练型 cache runner | 需要重建神经专家 cache 时；主线只建议 DLinear/PatchTST | `tests/test_quitobench_dlinear_expert_cache.py` |
| `tools/quitobench_expert_cache_audit.py` | 只读 expert cache schema/common-window 审计 | 任何 expert cache 比较前 | `tests/test_quitobench_expert_cache_audit.py` |
| `tools/quitobench_oracle_target_audit.py` | common-window true uniform / oracle target audit | 已有三专家 cache 后，进入 gate 前 | `tests/test_quitobench_oracle_target_audit.py` |
| `tools/quitobench_normalized_oracle_audit.py` | normalized-scale oracle target audit | raw-scale audit 后、gate 前；重建 Quito train-segment scaler 后评分 | `tests/test_quitobench_normalized_oracle_audit.py` |
| `tools/quitobench_visual_encoder_adapter_smoke.py` | visual embedding cache smoke | 重新生成与 canonical registry 对齐的 embedding cache 时 | `tests/test_quitobench_visual_encoder_adapter_smoke.py` |
| `tools/quitobench_common.py` | 低风险公共 helper | 被 audit/comparison/oracle 工具复用，不作为 CLI 运行 | `tests/test_quitobench_common.py` |
| `tools/workspace_move_manifest_check.py` | workspace move manifest 只读检查 | 每次移动/归档前后 | `tests/test_workspace_move_manifest_check.py` |

主线推荐调用顺序：

```text
window_registry
registry_subset
registry_audit
light_proxy
imageization
expert_cache
expert_cache_audit
oracle_target_audit
normalized_oracle_audit
visual_embedding_cache
gate_smoke
```

当前尚未实现正式 `gate_smoke` 入口，不应临时从历史 comparison 脚本拼接替代。

## 2. Reference / Diagnostic 入口

这些脚本可以保留，用于复查事实或诊断，但不应作为新主线默认入口。

| 脚本 | 角色 | 注意事项 |
| --- | --- | --- |
| `tools/quitobench_sufficiency_audit.py` | Stage 0 数据充分性审计 | 已有输出通常够用，除非数据版本变化 |
| `tools/quitobench_item_stl_quality_audit.py` | item 级 full-length STL/Quito quality audit | 耗时，复跑前确认必要性 |
| `tools/benchmark_quito_stl_quality.py` | STL/evaluate_series runtime benchmark | 只用于估时 |
| `tools/quitobench_channel_stl_codebook_validation.py` | 通道级 STL/codebook 验证 | 路线 2 解释用，不替代官方 cluster |
| `tools/quitobench_official_cluster_codebook.py` | 官方 cluster codebook 反推 | codebook 已固化，按需复查 |
| `tools/quitobench_official_cluster_semantics.py` | 官方 cluster 经验语义画像 | 解释辅助，不作为 router 标签 |
| `tools/quitobench_expert_prediction_diagnostics.py` | 预测/target/error scale 诊断 | 排查 PatchTST/DLinear 尺度异常时使用 |
| `tools/quitobench_proxy_imageization_latency.py` | proxy + imageization latency sweep | latency 结论需要干净机器环境重测 |
| `tools/quitobench_quito_native_sanity.py` | Quito native model/dataset sanity | 交互式慎用，不替代官方 full evaluation |
| `tools/quitobench_quito_native_utils.py` | native sanity 纯函数 helper | 被 native sanity 测试覆盖 |

## 3. Deprecated / Compatibility 入口

| 脚本 | 状态 | 处理建议 |
| --- | --- | --- |
| `tools/quitobench_dlinear_expert_cache.py` | compatibility wrapper | 保留，避免旧日志命令失效；新命令使用 `quitobench_framework_expert_cache.py` |
| `tools/quitobench_expert_cache_comparison.py` | canonical-needs-fix / legacy comparison | 可复用部分函数；不要把 `uniform_mse_proxy` 当 true ensemble 指标 |

`quitobench_expert_cache_comparison.py` 当前仍有价值：

- `load_error_tables`
- common-window filtering
- `build_true_uniform_ensemble_metrics`

但它的 CLI 默认仍指向旧 `h192/p96` comparison 口径。后续如果继续使用，应先改默认 sample set 或只把纯函数迁入公共 helper。

## 4. 不建议直接运行的旧分支

不要从已迁移到 `_deprecated` / `_scratch` 的 outputs 反推新命令继续跑：

- Stage 1.4d training stability diagnostics。
- Stage 1.4e wrapper/global scaler。
- early budget calibration。
- raw `stage14g_b` sparse sanity。
- TSMixer 扩展分支。
- 任意 `*_smoke*` 输出作为结论。

这些分支只能作为历史诊断证据。

## 5. 重复逻辑合并建议

已新增 `tools/quitobench_common.py`，目前只迁入低风险纯函数。后续仍应避免一次性大重构：

| 重复逻辑 | 当前位置 | 建议 |
| --- | --- | --- |
| prediction column discovery | `expert_cache_audit.py`、`expert_cache_comparison.py`、`oracle_target_audit.py`、`lightweight_expert_cache.py` | 已迁入 `prediction_columns()`；`lightweight_expert_cache.py` 暂未切换 |
| manifest loading/writing | 多个 expert/cache/audit 脚本 | 已迁入 `load_json_manifest()` / `write_json_manifest()`，先用于 audit/comparison/oracle |
| required-column and key checks | `expert_cache_audit.py`、`oracle_target_audit.py`、`expert_cache_comparison.py` | 已迁入 `require_columns()` / `ensure_unique_key()` |
| cache table loading | `expert_cache_audit.py`、`oracle_target_audit.py`、`expert_cache_comparison.py` | 暂不抽 `load_expert_cache_tables()`，因为 audit/comparison/oracle 的返回语义不同 |
| common-window filtering | `expert_cache_comparison.py`、`oracle_target_audit.py` | 已迁入 `filter_common_expert_windows()` |
| registry validation | `lightweight_expert_cache.py`、`framework_expert_cache.py`、`window_registry.py` | 先不要强抽，接口仍有差异 |
| history/target extraction | `lightweight_expert_cache.py`、`framework_expert_cache.py` | 先保持现状，避免影响标准化路径 |

重构顺序建议：

1. 先抽不依赖 Quito/raw data 的纯函数。
2. 再抽 cache schema helper。
3. 最后才考虑 history/target extraction 和 standardizer。

## 6. 当前不要做的事

- 不要删除兼容入口。
- 不要一次性把所有脚本改成 import common helper。
- 不要把 `stage14f` reference 和 current `official_align` cache 移到 `_deprecated`。
- 不要在 normalized/raw 指标未补齐前继续解释 PatchTST/DLinear/SNaive 排序为论文复现结论。
