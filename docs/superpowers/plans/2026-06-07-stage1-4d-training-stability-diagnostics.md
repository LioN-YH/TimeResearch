# Stage 1.4d Training Stability Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose PatchTST and TSMixer training instability before expanding the expert pool or running OOF cache.

**Architecture:** Add minimal CLI/config switches to the existing framework expert runner so RevIN, dropout, weight decay, and model capacity are explicit in manifests. Then run a small 20k stratified diagnostic matrix for PatchTST and TSMixer and compare each run against `seasonal_naive` plus the stable DLinear reference where useful.

**Tech Stack:** Python, pytest, conda environment `quito`, PyTorch/CUDA, existing Stage 1.4 expert cache and comparison scripts.

---

## Files

Modify:

- `tools/quitobench_framework_expert_cache.py`
  - Add CLI flags: `--no-revin`, `--weight-decay`, `--dropout`, `--fc-dropout`, `--head-dropout`.
  - Ensure config dataclasses and manifest config reflect user-supplied values.

- `tests/test_quitobench_dlinear_expert_cache.py`
  - Add tests proving the config dataclasses can represent RevIN off and dropout/weight decay overrides.
  - Add tests for parser default and `--no-revin` behavior without running full training.

Create:

- `experiment_logs/2026-06-07_1626_stage1_4d_training_stability_diagnostics.md`
  - Records diagnostic matrix, outputs, metrics, and next decision.

Modify after experiments:

- `experiment_logs/实验日志总览.md`
  - Adds Stage 1.4d row.

Generated outputs:

- PatchTST 20k diagnostic expert caches.
- TSMixer 20k diagnostic expert caches.
- Stage 1.4d comparison outputs.

---

### Task 1: Add Runner Parameter Tests

- [ ] **Step 1: Write failing tests**

Append to `tests/test_quitobench_dlinear_expert_cache.py`:

```python
from tools.quitobench_framework_expert_cache import parse_args


def test_patchtst_config_can_disable_revin_and_override_regularization() -> None:
    config = PatchTSTExpertConfig(
        revin=False,
        dropout=0.2,
        fc_dropout=0.15,
        head_dropout=0.05,
        weight_decay=0.01,
    )

    assert config.revin is False
    assert config.dropout == pytest.approx(0.2)
    assert config.fc_dropout == pytest.approx(0.15)
    assert config.head_dropout == pytest.approx(0.05)
    assert config.weight_decay == pytest.approx(0.01)


def test_tsmixer_config_can_disable_revin_and_override_dropout() -> None:
    config = TSMixerExpertConfig(revin=False, dropout=0.2, weight_decay=0.01)

    assert config.revin is False
    assert config.dropout == pytest.approx(0.2)
    assert config.weight_decay == pytest.approx(0.01)


def test_parse_args_exposes_stage14d_diagnostic_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--expert-model",
            "patchtst",
            "--no-revin",
            "--weight-decay",
            "0.01",
            "--dropout",
            "0.2",
            "--fc-dropout",
            "0.15",
            "--head-dropout",
            "0.05",
        ],
    )

    args = parse_args()

    assert args.revin is False
    assert args.weight_decay == pytest.approx(0.01)
    assert args.dropout == pytest.approx(0.2)
    assert args.fc_dropout == pytest.approx(0.15)
    assert args.head_dropout == pytest.approx(0.05)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py::test_parse_args_exposes_stage14d_diagnostic_flags -q
```

Expected: failure because `parse_args` does not expose `revin`, `weight_decay`, or dropout override flags yet.

### Task 2: Implement Runner Flags

- [ ] **Step 1: Add argparse flags**

Modify `tools/quitobench_framework_expert_cache.py` in `parse_args()`:

```python
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--fc-dropout", type=float, default=None)
parser.add_argument("--head-dropout", type=float, default=None)
parser.add_argument("--revin", dest="revin", action="store_true", default=True)
parser.add_argument("--no-revin", dest="revin", action="store_false")
```

- [ ] **Step 2: Wire flags into configs**

Use the parsed values when building model configs:

```python
config = PatchTSTExpertConfig(
    expert_set_id=args.expert_set_id,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    patch_len=args.patch_len,
    stride=args.stride,
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_heads=args.n_heads,
    e_layers=args.e_layers,
    dropout=args.dropout if args.dropout is not None else PatchTSTExpertConfig.dropout,
    fc_dropout=args.fc_dropout if args.fc_dropout is not None else PatchTSTExpertConfig.fc_dropout,
    head_dropout=args.head_dropout if args.head_dropout is not None else PatchTSTExpertConfig.head_dropout,
    revin=args.revin,
)
```

Use equivalent wiring for DLinear and TSMixer.

- [ ] **Step 3: Run targeted tests and verify GREEN**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py::test_parse_args_exposes_stage14d_diagnostic_flags tests/test_quitobench_dlinear_expert_cache.py::test_patchtst_config_can_disable_revin_and_override_regularization tests/test_quitobench_dlinear_expert_cache.py::test_tsmixer_config_can_disable_revin_and_override_dropout -q
```

Expected: all 3 pass.

- [ ] **Step 4: Run full framework expert tests**

Run:

```bash
conda run -n quito python -m pytest tests/test_quitobench_dlinear_expert_cache.py -q
```

Expected: all tests in the module pass.

- [ ] **Step 5: Commit runner flag change**

Run:

```bash
git add tools/quitobench_framework_expert_cache.py tests/test_quitobench_dlinear_expert_cache.py docs/superpowers/plans/2026-06-07-stage1-4d-training-stability-diagnostics.md
git commit -m "feat: expose expert runner diagnostic training flags"
```

Expected: one commit.

### Task 3: Run PatchTST 20k Diagnostics

- [ ] **Step 1: Run PatchTST lower learning rate**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model patchtst --stratified-rows 20000 --epochs 5 --batch-size 128 --learning-rate 0.0003 --expert-set-id patchtst_v1__stratified_20k_cuda_e5_lr3e4 --device cuda
```

Expected: 20k cache completes.

- [ ] **Step 2: Run PatchTST smaller model**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model patchtst --stratified-rows 20000 --epochs 5 --batch-size 128 --learning-rate 0.0003 --d-model 64 --d-ff 128 --e-layers 1 --n-heads 4 --expert-set-id patchtst_v1__stratified_20k_cuda_e5_small_lr3e4 --device cuda
```

Expected: 20k cache completes.

- [ ] **Step 3: Run PatchTST RevIN off lower learning rate**

Run:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model patchtst --stratified-rows 20000 --epochs 5 --batch-size 128 --learning-rate 0.0003 --no-revin --expert-set-id patchtst_v1__stratified_20k_cuda_e5_lr3e4_no_revin --device cuda
```

Expected: 20k cache completes.

### Task 4: Run TSMixer 20k Diagnostics

- [ ] **Step 1: Run TSMixer lower learning rate**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model tsmixer --stratified-rows 20000 --epochs 5 --batch-size 128 --learning-rate 0.0003 --num-blocks 2 --d-ff 64 --norm-type layer --expert-set-id tsmixer_v1__stratified_20k_cuda_e5_lr3e4 --device cuda
```

Expected: 20k cache completes.

- [ ] **Step 2: Run TSMixer RevIN off lower learning rate**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n quito python tools/quitobench_framework_expert_cache.py --expert-model tsmixer --stratified-rows 20000 --epochs 5 --batch-size 128 --learning-rate 0.0003 --num-blocks 2 --d-ff 64 --norm-type layer --no-revin --expert-set-id tsmixer_v1__stratified_20k_cuda_e5_lr3e4_no_revin --device cuda
```

Expected: 20k cache completes.

### Task 5: Compare and Log Stage 1.4d

- [ ] **Step 1: Compare PatchTST lower learning rate**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_20k_cuda_e5_lr3e4 --required-experts seasonal_naive,dlinear_quito,patchtst_quito --comparison-id stage1_4d_patchtst_20k_e5_lr3e4__seasonal_naive_dlinear
```

Expected: comparison completes with 20k common windows.

- [ ] **Step 2: Compare PatchTST smaller model**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_20k_cuda_e5_small_lr3e4 --required-experts seasonal_naive,dlinear_quito,patchtst_quito --comparison-id stage1_4d_patchtst_20k_e5_small_lr3e4__seasonal_naive_dlinear
```

Expected: comparison completes with 20k common windows.

- [ ] **Step 3: Compare PatchTST RevIN off**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/patchtst_v1__stratified_20k_cuda_e5_lr3e4_no_revin --required-experts seasonal_naive,dlinear_quito,patchtst_quito --comparison-id stage1_4d_patchtst_20k_e5_lr3e4_no_revin__seasonal_naive_dlinear
```

Expected: comparison completes with 20k common windows.

- [ ] **Step 4: Compare TSMixer lower learning rate**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_20k_cuda_e5_lr3e4 --required-experts seasonal_naive,dlinear_quito,tsmixer_quito --comparison-id stage1_4d_tsmixer_20k_e5_lr3e4__seasonal_naive_dlinear
```

Expected: comparison completes with 20k common windows.

- [ ] **Step 5: Compare TSMixer RevIN off**

Run:

```bash
conda run --no-capture-output -n quito python tools/quitobench_expert_cache_comparison.py --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/lightweight_v1__seasonal_naive_full --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/dlinear_v1__stratified_50k_cuda_e5 --cache-dir outputs/vision_ts_routing/expert_predictions/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/tsmixer_v1__stratified_20k_cuda_e5_lr3e4_no_revin --required-experts seasonal_naive,dlinear_quito,tsmixer_quito --comparison-id stage1_4d_tsmixer_20k_e5_lr3e4_no_revin__seasonal_naive_dlinear
```

Expected: comparison completes with 20k common windows.

- [ ] **Step 6: Write log**

Create `experiment_logs/2026-06-07_1626_stage1_4d_training_stability_diagnostics.md` with command matrix, training losses, comparison summaries, and decision.

- [ ] **Step 7: Update overview and commit**

Run:

```bash
git add experiment_logs/2026-06-07_1626_stage1_4d_training_stability_diagnostics.md experiment_logs/实验日志总览.md
git commit -m "docs: report stage 1.4d training stability diagnostics"
```

Expected: one commit with logs.
