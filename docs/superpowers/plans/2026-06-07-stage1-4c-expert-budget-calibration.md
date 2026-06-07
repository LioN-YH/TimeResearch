# Stage 1.4c Expert Budget Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Stage 1.4c training-budget calibration for DLinear, PatchTST, and TSMixer on a 50k stratified QuitoBench window sample.

**Architecture:** Reuse the existing `tools/quitobench_framework_expert_cache.py` runner to generate one expert cache per model/epoch setting, then reuse `tools/quitobench_expert_cache_comparison.py` to compare each epoch group against the full `seasonal_naive` baseline. No router, gate, visual encoder, OOF cache, or Quito upstream edits are introduced.

**Tech Stack:** Python, conda environment `quito`, PyTorch/CUDA, pandas/parquet cache files, existing Stage 1.4 expert cache scripts.

---

## Files

Modify:

- `experiment_logs/2026-06-07_1612_stage1_4c_expert_budget_calibration.md`
  - Records commands, GPU assignment, outputs, elapsed time, and conclusions.

- `experiment_logs/实验日志总览.md`
  - Adds one Stage 1.4c tracking row after the experiment has concrete results.

Read-only:

- `tools/quitobench_framework_expert_cache.py`
- `tools/quitobench_expert_cache_comparison.py`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full/`

Generated outputs:

- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e1/`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e1/`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e1/`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5/`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e5/`
- `outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e5/`
- `outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer/`
- `outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer/`

---

### Task 1: Preflight

- [ ] **Step 1: Confirm git status**

Run:

```bash
git status --short
```

Expected: clean or only intentional plan/log files.

- [ ] **Step 2: Confirm CUDA visibility**

Run:

```bash
conda run -n quito python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
```

Expected: CUDA available and at least one visible GPU.

- [ ] **Step 3: Confirm seasonal naive baseline exists**

Run:

```bash
ls outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full/errors.parquet
```

Expected: path exists.

### Task 2: Run 50k Epoch 1 Caches

- [ ] **Step 1: Run DLinear e1 on GPU 0**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model dlinear --stratified-rows 50000 --epochs 1 --batch-size 128 --expert-set-id dlinear_v1__stratified_50k_cuda_e1 --device cuda
```

Expected: output directory created, `windows=50000`, `prediction_rows=50000`, and nonzero `train_windows`.

- [ ] **Step 2: Run PatchTST e1 on GPU 1**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model patchtst --stratified-rows 50000 --epochs 1 --batch-size 128 --expert-set-id patchtst_v1__stratified_50k_cuda_e1 --device cuda
```

Expected: output directory created, `windows=50000`, `prediction_rows=50000`, and nonzero `train_windows`.

- [ ] **Step 3: Run TSMixer e1 on GPU 2**

Run:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model tsmixer --stratified-rows 50000 --epochs 1 --batch-size 128 --expert-set-id tsmixer_v1__stratified_50k_cuda_e1 --device cuda --num-blocks 2 --d-ff 64 --norm-type layer
```

Expected: output directory created, `windows=50000`, `prediction_rows=50000`, and nonzero `train_windows`.

### Task 3: Compare 50k Epoch 1

- [ ] **Step 1: Run e1 comparison**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e1 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e1 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e1 --required-experts seasonal_naive,dlinear_quito,patchtst_quito,tsmixer_quito --comparison-id budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer
```

Expected: `common_windows=50000` and four experts.

### Task 4: Run 50k Epoch 5 Caches

- [ ] **Step 1: Run DLinear e5 on GPU 0**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model dlinear --stratified-rows 50000 --epochs 5 --batch-size 128 --expert-set-id dlinear_v1__stratified_50k_cuda_e5 --device cuda
```

Expected: output directory created with 50k windows.

- [ ] **Step 2: Run PatchTST e5 on GPU 1**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model patchtst --stratified-rows 50000 --epochs 5 --batch-size 128 --expert-set-id patchtst_v1__stratified_50k_cuda_e5 --device cuda
```

Expected: output directory created with 50k windows.

- [ ] **Step 3: Run TSMixer e5 on GPU 2**

Run:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model tsmixer --stratified-rows 50000 --epochs 5 --batch-size 128 --expert-set-id tsmixer_v1__stratified_50k_cuda_e5 --device cuda --num-blocks 2 --d-ff 64 --norm-type layer
```

Expected: output directory created with 50k windows.

### Task 5: Compare 50k Epoch 5

- [ ] **Step 1: Run e5 comparison**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_50k_cuda_e5 --required-experts seasonal_naive,dlinear_quito,patchtst_quito,tsmixer_quito --comparison-id budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer
```

Expected: `common_windows=50000` and four experts.

### Task 6: Write Experiment Log

- [ ] **Step 1: Summarize manifests and comparisons**

Read:

```bash
cat outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer/comparison_summary.csv
cat outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e1__seasonal_naive_dlinear_patchtst_tsmixer/expert_metrics.csv
cat outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer/comparison_summary.csv
cat outputs/vision_ts_routing/expert_comparisons/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/budget_calibration_50k_e5__seasonal_naive_dlinear_patchtst_tsmixer/expert_metrics.csv
```

Expected: e1/e5 summaries available.

- [ ] **Step 2: Create Stage 1.4c experiment log**

Create `experiment_logs/2026-06-07_1612_stage1_4c_expert_budget_calibration.md` with:

- purpose;
- command matrix;
- GPU assignment;
- train/valid/test counts;
- model parameters;
- e1/e5 comparison results;
- conclusion on whether to run e20;
- conclusion on whether to proceed to OOF or model re-selection.

- [ ] **Step 3: Update global log overview**

Append one row to `experiment_logs/实验日志总览.md` describing Stage 1.4c status and next step.

- [ ] **Step 4: Commit docs and logs**

Run:

```bash
git add docs/superpowers/plans/2026-06-07-stage1-4c-expert-budget-calibration.md experiment_logs/2026-06-07_1612_stage1_4c_expert_budget_calibration.md experiment_logs/实验日志总览.md
git commit -m "docs: report stage 1.4c expert budget calibration"
```

Expected: one commit containing the plan and experiment logs.
