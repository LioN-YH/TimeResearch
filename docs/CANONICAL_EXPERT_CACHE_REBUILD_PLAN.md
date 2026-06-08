# 规范三专家 Cache 重建计划

> **给后续 agent/执行者的要求：** 如果按本文逐项实施，请使用 `superpowers:executing-plans` 这类逐步执行流程。任务项使用 checkbox（`- [ ]`）语法，便于跟踪状态。

**目标：** 重建 SNaive / DLinear / PatchTST 三个专家 cache，使三者严格使用同一份 sampled registry，并产出一份干净、可审计的 expert matrix。

**核心架构：** 先把固定抽样结果物化为一个真正的 `window_registry` 目录。然后让三个专家 cache builder 都读取同一个 `window_index.csv`，并且不再传入 `--stratified-rows` 或 `--max-rows`。最后运行 cache audit 和 oracle target audit，验证 key 唯一性、horizon 对齐、共同窗口数、raw-scale 指标，并判断在 gate 训练前是否必须补充 normalized-scale audit。

**技术栈：** Python、pandas/parquet、PyTorch、Quito model wrappers、现有 `conda run -n quito` 环境。

---

## 0. 当前决策

应该重建三专家 cache。

原因：

- 当前 SNaive / DLinear / PatchTST 三份 50k cache 不是同一批 50k windows。
- 当前三者共同窗口只有 23,456 个。
- 当前比较仍然只是 raw-scale。
- PatchTST / DLinear / SNaive 的排序相对 Quito 论文预期仍然可疑。

重要修正：

不要分别给 SNaive、DLinear、PatchTST 传 `--stratified-rows 50000` 来重建。这样会重复原来的问题：三者各自独立抽样，不能保证窗口集合一致。正确做法是：

1. 先创建一个固定 sampled registry 目录。
2. 三个专家都针对这个固定目录运行，不再额外抽样。
3. 明确确认每份 cache 的 `physical_window_id` 集合完全一致。

## 1. 目标产物

第一轮推荐的干净重建产物：

```text
base registry:
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506

new fixed sampled registry:
outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

new expert output root:
outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1

expert set ids:
seasonal_naive_period6__matrix50k_v1
dlinear__matrix50k_v1_e20_std
patchtst__matrix50k_v1_e20_std

audit outputs:
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
```

为什么先做 50k：

- 和当前 cache 的大致规模一致，便于对照。
- 对 sanity run 来说规模足够。
- 在确认 pipeline 正确前，避免直接跳到 220,755-window 的 full sparse training。

50k clean matrix 通过 sanity 后，可以再选择性地对完整 `220,755` stride288 registry 做第二轮 full sparse rebuild。

## 2. Task A：新增固定 Registry 子集工具

**文件：**

- 新增：`tools/quitobench_registry_subset.py`
- 新增：`tests/test_quitobench_registry_subset.py`
- 更新：`docs/TOOL_ENTRYPOINTS.md`
- 更新：`docs/ARTIFACT_INVENTORY.md`

目的：

把一次抽样结果物化为一份一等公民的 registry 目录。输出的 `window_index.csv` 必须在每一行设置新的 `sample_set_id`，同时保留：

- `physical_window_id`
- `window_id`
- `base_registry_id`
- `subset`
- `split`
- `item_id`
- `channel`
- `official_tsf_cell`
- 所有 window boundary columns

要求的 CLI：

```bash
conda run -n quito python tools/quitobench_registry_subset.py \
  --input-registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506 \
  --output-registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --sample-set-id qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --target-rows 50000 \
  --stratify-cols split,subset,official_tsf_cell \
  --random-seed 20260608
```

要求行为：

- 如果输入 registry 中 `physical_window_id` 不唯一，直接失败。
- 如果输出路径已存在，除非显式传入 `--overwrite`，否则失败。
- 同一 seed 下抽样必须确定性一致。
- 输出：
  - `window_index.csv`
  - `manifest.json`
  - 如果输入目录存在 `config.yml`，则复制到输出目录

Manifest 必须包含：

```json
{
  "stage": "canonical_expert_matrix_registry_subset",
  "input_registry_dir": "...",
  "sample_set_id": "qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1",
  "base_sample_set_id": "qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506",
  "target_rows": 50000,
  "selected_rows": 50000,
  "stratify_cols": ["split", "subset", "official_tsf_cell"],
  "random_seed": 20260608,
  "split_window_counts": {},
  "subset_window_counts": {},
  "cell_window_counts": {},
  "unique_items": 0,
  "unique_channels": []
}
```

测试命令：

```bash
conda run -n quito python -m pytest tests/test_quitobench_registry_subset.py tests/test_quitobench_registry_audit.py -q
```

## 3. Task B：生成并审计固定 Registry

运行：

```bash
conda run -n quito python tools/quitobench_registry_subset.py \
  --input-registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506 \
  --output-registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --sample-set-id qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --target-rows 50000 \
  --stratify-cols split,subset,official_tsf_cell \
  --random-seed 20260608
```

然后审计：

```bash
conda run -n quito python tools/quitobench_registry_audit.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --output-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/audit
```

验收标准：

- `window_index.csv` 恰好有 50,000 行。
- `physical_window_id` 唯一。
- `sample_set_id` 只有一个取值：`qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1`。
- split / subset / cell 的计数在预期 group 中均非零。

## 4. Task C：在固定 Registry 上重建 SNaive

运行：

```bash
conda run -n quito python tools/quitobench_lightweight_expert_cache.py \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --output-root outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --expert-set-id seasonal_naive_period6__matrix50k_v1 \
  --expert-ids seasonal_naive \
  --seasonal-period-override 6
```

不要传：

- `--stratified-rows`
- `--max-rows`

验收标准：

- `prediction_rows == 50000`
- `error_rows == 50000`
- `expert_id == seasonal_naive`
- 抽查窗口中，prediction 每 6 个 horizon 重复。

## 5. Task D：在固定 Registry 上重建 DLinear

第一轮干净运行建议：

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model dlinear \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --output-root outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --expert-set-id dlinear__matrix50k_v1_e20_std \
  --epochs 20 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 0.001 \
  --weight-decay 0.0 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.00001 \
  --device cuda \
  --progress-every 100
```

不要传：

- `--stratified-rows`
- `--max-rows`

验收标准：

- `prediction_rows == 50000`
- `error_rows == 50000`
- manifest 中 `standardization.enabled == true`
- manifest 中 `standardization.scope == quito_timeseries_dataset_train_segment`
- train / eval / pred horizon 都是 48。

## 6. Task E：在固定 Registry 上重建 PatchTST

第一轮干净运行建议：

```bash
conda run -n quito python tools/quitobench_framework_expert_cache.py \
  --expert-model patchtst \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --output-root outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --expert-set-id patchtst__matrix50k_v1_e20_std \
  --epochs 20 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 0.0001 \
  --weight-decay 0.0 \
  --train-set-standardize \
  --drop-last \
  --scheduler cosine \
  --eta-min 0.000001 \
  --patch-len 16 \
  --stride 8 \
  --d-model 128 \
  --d-ff 256 \
  --n-heads 4 \
  --e-layers 2 \
  --dropout 0.05 \
  --fc-dropout 0.05 \
  --head-dropout 0.0 \
  --device cuda \
  --progress-every 100
```

不要传：

- `--stratified-rows`
- `--max-rows`

验收标准：

- `prediction_rows == 50000`
- `error_rows == 50000`
- manifest 中 `source_model == quito.models.patchtst.PatchTST`
- manifest 确认启用了 Quito train-set standardization。
- `yhat_*` 中没有 NaN 或 inf。

## 7. Task F：审计三份干净 Cache

运行：

```bash
conda run -n quito python tools/quitobench_expert_cache_audit.py \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/seasonal_naive_period6__matrix50k_v1 \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/dlinear__matrix50k_v1_e20_std \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/patchtst__matrix50k_v1_e20_std \
  --output-dir outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1
```

验收标准：

- `common_prediction_windows == 50000`
- `common_error_windows == 50000`
- 所有 horizon 和 manifest 一致。
- 所有 prediction / error keys 唯一。

如果 common windows 不是严格 50,000，立即停止。不要继续进入 gate / visual work。

## 8. Task G：运行 Oracle Target Audit

运行：

```bash
conda run -n quito python tools/quitobench_oracle_target_audit.py \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/seasonal_naive_period6__matrix50k_v1 \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/dlinear__matrix50k_v1_e20_std \
  --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/patchtst__matrix50k_v1_e20_std \
  --registry-dir outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --audit-id matrix50k_v1 \
  --output-root outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1 \
  --progress-every 5000
```

验收标准：

- `num_common_windows == 50000`
- 输出 best fixed、true uniform、oracle top1、top1 rates。
- 对 raw-scale 解释进行明确标注。

重要：

这仍然只是 raw-scale。正式训练 gate 前，必须补充 normalized-scale oracle audit，或者扩展 expert cache 输出 normalized-scale errors。

## 9. Task H：Normalized-Scale Audit 要求

不要把 raw oracle audit 当成最终结论。下一步必须做以下两种工具改动之一：

方案 A：

- 扩展 expert cache builders，使其同时输出 raw 和 normalized prediction / error tables。

方案 B：

- 新增一个 normalized oracle audit 工具，重建 Quito train-segment scaler，并在 normalized scale 上重新计算已有 predictions / targets 的指标。

建议优先实现方案 B，因为它不会改变当前 expert cache 生成逻辑，更适合在我们仍在排查 ranking inversion 时使用。

潜在输出：

```text
outputs/vision_ts_routing/oracle_audit/.../matrix50k_v1_normalized/
```

## 10. 停止 / 继续标准

只有在满足以下条件时，才继续 visual embedding 和 gate smoke：

- 三份干净 cache 恰好有 50,000 个 common windows；
- DLinear / PatchTST manifests 确认使用 train-set standardization；
- SNaive period=6 抽查通过；
- raw 和 normalized audits 都没有显著 scale / pathology bug；
- PatchTST 排名异常至少能由配置或训练预算解释，而不是由数据泄漏或指标不匹配导致。

遇到以下情况则停止并调试 expert rebuild：

- common windows 少于 50,000；
- 任一 cache 使用了额外 sampling；
- 任一 cache 存在 target leakage 风险；
- 任一 neural expert 产生 NaN / inf predictions；
- normalized-scale metrics 和 raw-scale metrics 出现无法解释的矛盾。

## 11. 执行后的文档更新

重建完成后，更新：

- `docs/STAGE_RESET_SUMMARY.md`
- `docs/NEXT_STAGE_CANONICAL_PROTOCOL.md`
- `docs/ARTIFACT_INVENTORY.md`
- `docs/TOOL_ENTRYPOINTS.md`

记录：

- 精确 registry id；
- 精确 expert set ids；
- 使用的 GPU / device；
- train epoch / batch / lr 配置；
- cache audit common windows；
- raw oracle audit 结果；
- normalized audit 状态；
- 是否允许恢复 visual / gate work。

## 12. 2026-06-08 执行记录

已完成 `matrix50k_v1` clean rebuild：

- fixed registry: `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1`
- selected rows: `50000`
- split counts: train `30134`，valid `8192`，test `11674`
- subset counts: hour `31617`，min `18383`
- min split/subset/cell windows: `9`
- SNaive: `seasonal_naive_period6__matrix50k_v1`
- DLinear: `dlinear__matrix50k_v1_e20_std`
- PatchTST: `patchtst__matrix50k_v1_e20_std`
- GPU/device: DLinear 和 PatchTST 均使用 `CUDA_VISIBLE_DEVICES=0` / `cuda:0`
- neural config: epochs `20`，batch `256`，eval batch `512`，Quito train-set standardization enabled
- SNaive period=6 spot check: pass
- all `yhat_*` finite / no NaN: pass

Cache audit:

```text
outputs/vision_ts_routing/expert_cache_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
```

- `common_prediction_windows == 50000`
- `common_error_windows == 50000`
- all horizons match manifest: true
- all prediction/error keys unique: true

Raw-scale oracle audit:

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1/
```

- best fixed expert: `dlinear_quito`
- best fixed MSE/MAE: `2.626859e+11` / `45414.205961`
- true uniform MSE/MAE: `7.625250e+11` / `58412.098425`
- oracle top1 MSE/MAE: `2.197719e+11` / `36343.860462`
- top1 rates: SNaive `0.53038`，DLinear `0.33292`，PatchTST `0.13670`

Normalized-scale oracle audit:

```text
outputs/vision_ts_routing/oracle_audit/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506__matrix50k_v1/matrix50k_v1_normalized/
```

- tool: `tools/quitobench_normalized_oracle_audit.py`
- scaler: Quito train-segment subset/item/channel scaler
- common windows: `50000`
- best fixed expert by normalized MSE: `seasonal_naive`
- best fixed normalized MSE/MAE: `1.459632` / `0.510185`
- DLinear normalized MSE/MAE: `127.947743` / `0.525655`
- PatchTST normalized MSE/MAE: `736.788875` / `1.020283`
- oracle top1 normalized MSE/MAE: `0.786171` / `0.350153`

当前结论：

- clean 50k expert matrix 已经生成并通过 common-window 审计。
- 但 raw-scale 和 normalized-scale 排名出现明显分歧：raw best fixed 是 DLinear，normalized MSE best fixed 是 SNaive。
- normalized MSE 中 DLinear/PatchTST 明显被大误差放大，但 DLinear normalized MAE 与 SNaive 接近；这提示可能存在少数小 std window、outlier 或 metric-scale pathology。
- 因此暂不建议进入正式 visual/gate 训练。下一步应先排查 normalized error 分布、按 subset/cell/window std 分解异常，并确认是否需要 winsorized/MAE/Huber 或按官方 evaluate_series 口径补充审计。
