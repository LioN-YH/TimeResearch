# Stage 1.3a0 Visual Embedding Cache Smoke 实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行本计划。步骤使用 checkbox（`- [ ]`）追踪。

**目标：** 实现一个最小 visual embedding cache smoke，读取 Stage 1.2 的 `view_tensor [B,3,64,192]`，写出与 `physical_window_id` 对齐的 visual embedding cache。

**架构：** Stage 1.3a0 只验证视觉输入到 embedding cache 的接口，不接入专家训练和 router。脚本复用 Stage 1.2 已生成的 image tensor，运行一个 deterministic lightweight CNN adapter，并写出 `embeddings.parquet`、`embedding_index.csv`、`latency.csv` 和 `manifest.json` 到 `outputs/vision_ts_routing/visual_embeddings/{sample_set_id}/visual_embedding_cache_smoke_v1/`。本阶段不训练视觉 encoder，不实现 gate/router，不读取 expert errors，不修改正在运行的 Stage 1.4g-b expert runner。

**范围修正：** 本计划不是原先定义的 Stage 1.3a adapter comparison。它不比较 `per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding`，只作为这些路线之前的 cache/IO smoke。

**技术栈：** Python 3.11、pandas、numpy、torch、pytest、conda env `quito`。

---

## 文件结构

- 新增 `tools/quitobench_visual_encoder_adapter_smoke.py`
  - 定义 `VisualEncoderSmokeConfig`。
  - 定义 `TinyViewCnnEncoder`，用于 deterministic smoke，不作为正式视觉先验结论。
  - 定义 `load_stage12_view_tensor()`、`encode_view_tensor()`、`build_embedding_table()`、`write_visual_embedding_outputs()` 和 CLI。
  - manifest 记录 embedding shape、latency、device、输入 image protocol 和明确非目标。
- 新增 `tests/test_quitobench_visual_encoder_adapter_smoke.py`
  - 覆盖 encoder 输出 shape、确定性、embedding cache schema、manifest flags 和文件写出。
- 新增 `experiment_logs/2026-06-07_2255_stage1_3a0_visual_embedding_cache_smoke.md`
  - 记录执行命令、输入输出、验证结果、问题和下一步。
- 修改 `experiment_logs/实验日志总览.md`
  - 登记 Stage 1.3a0。

不得修改：

- `tools/quitobench_framework_expert_cache.py`
- `outputs/vision_ts_routing/expert_predictions/`
- `outputs/vision_ts_routing/window_registry/qb_h96_p48_quito_overlap_f4c3e571_stride288_b5cee506/`

这些路径属于并行运行中的 Stage 1.4g-b。

---

### 任务 1：Encoder Adapter 核心

**文件：**
- 新增：`tests/test_quitobench_visual_encoder_adapter_smoke.py`
- 新增：`tools/quitobench_visual_encoder_adapter_smoke.py`

- [ ] **步骤 1：写红灯测试**

创建 `tests/test_quitobench_visual_encoder_adapter_smoke.py`：

```python
from __future__ import annotations

import torch

from tools.quitobench_visual_encoder_adapter_smoke import (
    TinyViewCnnEncoder,
    VisualEncoderSmokeConfig,
    encode_view_tensor,
)


def test_tiny_view_cnn_encoder_outputs_embedding_shape() -> None:
    config = VisualEncoderSmokeConfig(embedding_dim=32)
    encoder = TinyViewCnnEncoder(config)
    view_tensor = torch.linspace(0.0, 1.0, steps=4 * 3 * 64 * 192, dtype=torch.float32).reshape(4, 3, 64, 192)

    embeddings = encoder(view_tensor)

    assert embeddings.shape == (4, 32)
    assert torch.isfinite(embeddings).all()


def test_encode_view_tensor_is_deterministic_for_same_weights() -> None:
    config = VisualEncoderSmokeConfig(embedding_dim=16, random_seed=123)
    view_tensor = torch.ones((3, 3, 64, 192), dtype=torch.float32)

    first, first_meta = encode_view_tensor(view_tensor, config=config, device="cpu")
    second, second_meta = encode_view_tensor(view_tensor, config=config, device="cpu")

    assert first.shape == (3, 16)
    assert torch.allclose(first, second)
    assert first_meta["encoder_id"] == "tiny_view_cnn_v1"
    assert second_meta["embedding_dim"] == 16
```

- [ ] **步骤 2：确认红灯**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期：失败，错误为 `ModuleNotFoundError: No module named 'tools.quitobench_visual_encoder_adapter_smoke'`。

- [ ] **步骤 3：实现最小 deterministic adapter**

创建 `tools/quitobench_visual_encoder_adapter_smoke.py`，包含：

```python
"""Stage 1.3a0：visual embedding cache smoke。

本脚本读取 Stage 1.2 view tensor，并写出 physical_window_id 对齐的
visual embedding cache。不训练视觉 encoder，不运行专家模型，不实现 router/gate。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_IMAGE_TENSOR_DIR = (
    ROOT
    / "outputs/vision_ts_routing/image_tensors"
    / "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/visual_embeddings"


@dataclass(frozen=True)
class VisualEncoderSmokeConfig:
    stage: str = "stage1_3a0_visual_embedding_cache_smoke"
    encoder_id: str = "tiny_view_cnn_v1"
    input_protocol_id: str = "view3_h64_w192_v1"
    input_view_dim: int = 3
    input_height: int = 64
    input_width: int = 192
    embedding_dim: int = 64
    random_seed: int = 20260607
    batch_size: int = 128


class TinyViewCnnEncoder(nn.Module):
    """只用于验证 visual embedding IO 的小型 deterministic adapter。"""

    def __init__(self, config: VisualEncoderSmokeConfig) -> None:
        super().__init__()
        torch.manual_seed(int(config.random_seed))
        self.net = nn.Sequential(
            nn.Conv2d(config.input_view_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, config.embedding_dim),
        )

    def forward(self, view_tensor: torch.Tensor) -> torch.Tensor:
        if view_tensor.ndim != 4:
            raise ValueError(f"view_tensor 必须是 [B,V,H,W]，当前 shape={tuple(view_tensor.shape)}")
        return self.net(view_tensor.to(dtype=torch.float32))


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def encode_view_tensor(
    view_tensor: torch.Tensor,
    config: VisualEncoderSmokeConfig | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, object]]:
    cfg = config or VisualEncoderSmokeConfig()
    torch_device = _select_device(device)
    encoder = TinyViewCnnEncoder(cfg).to(torch_device)
    encoder.eval()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(view_tensor.shape[0]), int(cfg.batch_size)):
            batch = view_tensor[start : start + int(cfg.batch_size)].to(torch_device)
            outputs.append(encoder(batch).detach().cpu())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed_seconds = time.perf_counter() - started
    embeddings = torch.cat(outputs, dim=0)
    meta = {
        "stage": cfg.stage,
        "encoder_id": cfg.encoder_id,
        "input_protocol_id": cfg.input_protocol_id,
        "device": str(torch_device),
        "embedding_dim": int(cfg.embedding_dim),
        "num_windows": int(embeddings.shape[0]),
        "elapsed_seconds": float(elapsed_seconds),
        "encoder_latency_ms_per_window": float(elapsed_seconds * 1000.0 / max(int(embeddings.shape[0]), 1)),
        "trains_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
    }
    return embeddings, meta
```

- [ ] **步骤 4：确认绿灯**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期：当前两个测试通过。

---

### 任务 2：Embedding Cache Schema 和输出写入

**文件：**
- 修改：`tests/test_quitobench_visual_encoder_adapter_smoke.py`
- 修改：`tools/quitobench_visual_encoder_adapter_smoke.py`

- [ ] **步骤 1：新增 embedding table 和 writer 红灯测试**

在测试文件中追加：

```python
import json
from pathlib import Path

import pandas as pd

from tools.quitobench_visual_encoder_adapter_smoke import (
    build_embedding_table,
    write_visual_embedding_outputs,
)


def _toy_image_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physical_window_id": ["w0", "w1", "w2"],
            "window_id": ["w0", "w1", "w2"],
            "base_registry_id": ["base_a", "base_a", "base_a"],
            "sample_set_id": ["sample_a", "sample_a", "sample_a"],
            "subset": ["hour", "hour", "min"],
            "split": ["train", "valid", "test"],
            "item_id": [1, 2, 3],
            "channel": ["ind_1", "ind_2", "ind_3"],
            "period": [24, 24, 144],
            "official_tsf_cell": ["highT_highS_highF", "highT_highS_lowF", "lowT_lowS_lowF"],
            "view_tensor_row": [0, 1, 2],
        }
    )


def test_build_embedding_table_keeps_physical_window_id_and_wide_columns() -> None:
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    table = build_embedding_table(_toy_image_index(), embeddings, encoder_id="tiny_view_cnn_v1")

    assert table["physical_window_id"].tolist() == ["w0", "w1", "w2"]
    assert table["sample_set_id"].nunique() == 1
    assert table["encoder_id"].unique().tolist() == ["tiny_view_cnn_v1"]
    assert table[["z_0", "z_1", "z_2", "z_3"]].shape == (3, 4)
    assert table["physical_window_id"].is_unique


def test_write_visual_embedding_outputs_writes_expected_files(tmp_path: Path) -> None:
    embeddings = torch.ones((3, 4), dtype=torch.float32)
    table = build_embedding_table(_toy_image_index(), embeddings, encoder_id="tiny_view_cnn_v1")
    manifest = {
        "stage": "stage1_3a0_visual_embedding_cache_smoke",
        "sample_set_id": "sample_a",
        "encoder_id": "tiny_view_cnn_v1",
        "embedding_dim": 4,
        "num_windows": 3,
        "trains_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
    }
    latency_rows = [{"device": "cpu", "batch_size": 3, "encoder_latency_ms_per_window": 0.1}]

    out_dir = write_visual_embedding_outputs(
        embedding_table=table,
        image_index=_toy_image_index(),
        latency_rows=latency_rows,
        manifest=manifest,
        output_root=tmp_path,
    )

    assert out_dir.parent.name == "sample_a"
    assert out_dir.name == "visual_embedding_cache_smoke_v1"
    assert (out_dir / "embeddings.parquet").exists()
    assert (out_dir / "embedding_index.csv").exists()
    assert (out_dir / "latency.csv").exists()
    assert (out_dir / "manifest.json").exists()
    loaded_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["implements_router"] is False
```

- [ ] **步骤 2：确认红灯**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期：失败，因为 `build_embedding_table()` 和 `write_visual_embedding_outputs()` 尚不存在。

- [ ] **步骤 3：实现 embedding table 和 writer**

在脚本中追加：

```python
def build_embedding_table(image_index: pd.DataFrame, embeddings: torch.Tensor, encoder_id: str) -> pd.DataFrame:
    if len(image_index) != int(embeddings.shape[0]):
        raise ValueError("image_index 行数必须等于 embeddings 行数")
    if not image_index["physical_window_id"].is_unique:
        raise ValueError("image_index physical_window_id 必须唯一")
    keep_cols = [
        "physical_window_id",
        "window_id",
        "base_registry_id",
        "sample_set_id",
        "subset",
        "split",
        "item_id",
        "channel",
        "period",
        "official_tsf_cell",
        "view_tensor_row",
    ]
    table = image_index[keep_cols].copy().reset_index(drop=True)
    table["encoder_id"] = str(encoder_id)
    emb_np = embeddings.detach().cpu().numpy()
    for idx in range(emb_np.shape[1]):
        table[f"z_{idx}"] = emb_np[:, idx].astype(np.float32)
    if table[["physical_window_id", "encoder_id"]].duplicated().any():
        raise ValueError("embedding table 存在重复 (physical_window_id, encoder_id)")
    return table


def write_visual_embedding_outputs(
    embedding_table: pd.DataFrame,
    image_index: pd.DataFrame,
    latency_rows: list[dict[str, object]],
    manifest: Mapping[str, object],
    output_root: Path,
) -> Path:
    sample_set_values = sorted(embedding_table["sample_set_id"].dropna().unique().tolist())
    if len(sample_set_values) != 1:
        raise ValueError(f"embedding_table 必须只包含一个 sample_set_id，当前为 {sample_set_values}")
    sample_set_id = str(sample_set_values[0])
    out_dir = output_root / sample_set_id / "visual_embedding_cache_smoke_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_table.to_parquet(out_dir / "embeddings.parquet", index=False)
    image_index.to_csv(out_dir / "embedding_index.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(out_dir / "latency.csv", index=False)
    manifest_to_write = dict(manifest)
    manifest_to_write["output_dir_name"] = f"{sample_set_id}/visual_embedding_cache_smoke_v1"
    manifest_to_write["output_files"] = {
        "embeddings": "embeddings.parquet",
        "embedding_index": "embedding_index.csv",
        "latency": "latency.csv",
        "manifest": "manifest.json",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_to_write, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir
```

- [ ] **步骤 4：确认绿灯**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期：Stage 1.3a0 当前测试全部通过。

---

### 任务 3：Stage 1.2 输出读取和 CLI Smoke

**文件：**
- 修改：`tools/quitobench_visual_encoder_adapter_smoke.py`
- 修改：`tests/test_quitobench_visual_encoder_adapter_smoke.py`

- [ ] **步骤 1：新增 Stage 1.2 loader 红灯测试**

在测试文件中追加：

```python
import numpy as np

from tools.quitobench_visual_encoder_adapter_smoke import load_stage12_view_tensor


def test_load_stage12_view_tensor_validates_index_and_tensor(tmp_path: Path) -> None:
    image_dir = tmp_path / "stage12"
    image_dir.mkdir()
    np.savez_compressed(image_dir / "view_tensor_sample.npz", view_tensor=np.zeros((2, 3, 64, 192), dtype=np.float32))
    pd.DataFrame(
        {
            "physical_window_id": ["w0", "w1"],
            "window_id": ["w0", "w1"],
            "base_registry_id": ["base_a", "base_a"],
            "sample_set_id": ["sample_a", "sample_a"],
            "subset": ["hour", "min"],
            "split": ["train", "test"],
            "item_id": [1, 2],
            "channel": ["ind_1", "ind_2"],
            "period": [24, 144],
            "official_tsf_cell": ["highT_highS_highF", "lowT_lowS_lowF"],
            "view_tensor_row": [0, 1],
        }
    ).to_csv(image_dir / "image_index.csv", index=False)
    (image_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "stage1_2_imageization_protocol_smoke",
                "sample_set_id": "sample_a",
                "image_protocol_id": "view3_h64_w192_v1",
                "view_tensor_semantics": "multi_view_not_rgb",
                "normalization": {"scope": "per_physical_window_id_history"},
                "tensor_shape": [2, 3, 64, 192],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    view_tensor, image_index, manifest = load_stage12_view_tensor(image_dir)

    assert view_tensor.shape == (2, 3, 64, 192)
    assert image_index["physical_window_id"].tolist() == ["w0", "w1"]
    assert manifest["view_tensor_semantics"] == "multi_view_not_rgb"
```

- [ ] **步骤 2：确认红灯**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py::test_load_stage12_view_tensor_validates_index_and_tensor -q
```

预期：失败，因为 `load_stage12_view_tensor()` 尚不存在。

- [ ] **步骤 3：实现 Stage 1.2 loader 和 manifest builder**

在脚本中追加：

```python
def load_stage12_view_tensor(image_tensor_dir: Path) -> tuple[torch.Tensor, pd.DataFrame, dict[str, object]]:
    tensor_path = image_tensor_dir / "view_tensor_sample.npz"
    index_path = image_tensor_dir / "image_index.csv"
    manifest_path = image_tensor_dir / "manifest.json"
    if not tensor_path.exists():
        raise FileNotFoundError(f"缺少 Stage 1.2 tensor：{tensor_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"缺少 Stage 1.2 index：{index_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 Stage 1.2 manifest：{manifest_path}")
    loaded = np.load(tensor_path)
    view_tensor = torch.tensor(loaded["view_tensor"], dtype=torch.float32)
    image_index = pd.read_csv(index_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(image_index) != int(view_tensor.shape[0]):
        raise ValueError("Stage 1.2 image_index 行数与 view_tensor 行数不一致")
    if not image_index["physical_window_id"].is_unique:
        raise ValueError("Stage 1.2 image_index physical_window_id 不唯一")
    if view_tensor.ndim != 4:
        raise ValueError(f"Stage 1.2 view_tensor 必须是 [B,V,H,W]，当前 shape={tuple(view_tensor.shape)}")
    return view_tensor, image_index, manifest


def build_visual_embedding_manifest(
    config: VisualEncoderSmokeConfig,
    image_tensor_dir: Path,
    stage12_manifest: Mapping[str, object],
    image_index: pd.DataFrame,
    encode_meta: Mapping[str, object],
    embedding_table: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    sample_set_values = sorted(image_index["sample_set_id"].dropna().unique().tolist())
    base_registry_values = sorted(image_index["base_registry_id"].dropna().unique().tolist())
    embedding_cols = [col for col in embedding_table.columns if col.startswith("z_")]
    return {
        "stage": config.stage,
        "encoder_id": config.encoder_id,
        "config": asdict(config),
        "input_image_tensor_dir": str(image_tensor_dir),
        "input_stage12_manifest": dict(stage12_manifest),
        "sample_set_id": sample_set_values[0] if len(sample_set_values) == 1 else sample_set_values,
        "base_registry_id": base_registry_values[0] if len(base_registry_values) == 1 else base_registry_values,
        "num_windows": int(len(image_index)),
        "embedding_dim": int(len(embedding_cols)),
        "embedding_format": "wide_columns",
        "embedding_columns": embedding_cols,
        "unique_physical_window_id": bool(image_index["physical_window_id"].is_unique),
        "view_tensor_semantics": stage12_manifest.get("view_tensor_semantics", "multi_view_not_rgb"),
        "normalization": stage12_manifest.get("normalization", {}),
        "device": encode_meta["device"],
        "encoder_latency_ms_per_window": encode_meta["encoder_latency_ms_per_window"],
        "trains_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
        "reads_expert_errors": False,
        "uses_future_target": False,
        "concurrent_stage14gb_safe": True,
    }
```

- [ ] **步骤 4：增加 CLI**

在脚本中追加：

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tensor-dir", type=Path, default=DEFAULT_IMAGE_TENSOR_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VisualEncoderSmokeConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )
    view_tensor, image_index, stage12_manifest = load_stage12_view_tensor(args.image_tensor_dir)
    print(f"[input] view_tensor_shape={list(view_tensor.shape)} rows={len(image_index)}")
    embeddings, encode_meta = encode_view_tensor(view_tensor, config=config, device=args.device)
    embedding_table = build_embedding_table(image_index, embeddings, encoder_id=config.encoder_id)
    latency_rows = [
        {
            "stage": config.stage,
            "encoder_id": config.encoder_id,
            "device": encode_meta["device"],
            "batch_size": int(config.batch_size),
            "num_windows": int(len(image_index)),
            "encoder_latency_ms_per_window": encode_meta["encoder_latency_ms_per_window"],
        }
    ]
    manifest = build_visual_embedding_manifest(
        config=config,
        image_tensor_dir=args.image_tensor_dir,
        stage12_manifest=stage12_manifest,
        image_index=image_index,
        encode_meta=encode_meta,
        embedding_table=embedding_table,
        args=args,
    )
    out_dir = write_visual_embedding_outputs(
        embedding_table=embedding_table,
        image_index=image_index,
        latency_rows=latency_rows,
        manifest=manifest,
        output_root=args.output_root,
    )
    print(f"[done] output={out_dir}")
    print(f"[done] embeddings_shape={[int(len(embedding_table)), int(config.embedding_dim)]}")
    print(f"[done] encoder_latency_ms_per_window={encode_meta['encoder_latency_ms_per_window']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行 CLI smoke**

运行：

```bash
conda run -n quito python tools/quitobench_visual_encoder_adapter_smoke.py \
  --image-tensor-dir outputs/vision_ts_routing/image_tensors/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e__stage1_2_smoke_v1 \
  --embedding-dim 64 \
  --batch-size 128 \
  --device cpu
```

预期输出包含：

```text
[input] view_tensor_shape=[288, 3, 64, 192] rows=288
[done] output=outputs/vision_ts_routing/visual_embeddings/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1
[done] embeddings_shape=[288, 64]
```

- [ ] **步骤 6：运行 Stage 1.3a0 全部测试**

运行：

```bash
conda run -n quito python -m pytest tests/test_quitobench_visual_encoder_adapter_smoke.py -q
```

预期：Stage 1.3a0 全部测试通过。

---

### 任务 4：验证、实验日志和总览

**文件：**
- 新增：`experiment_logs/2026-06-07_2255_stage1_3a0_visual_embedding_cache_smoke.md`
- 修改：`experiment_logs/实验日志总览.md`

- [ ] **步骤 1：运行聚焦回归测试**

运行：

```bash
conda run -n quito python -m pytest \
  tests/test_quitobench_visual_encoder_adapter_smoke.py \
  tests/test_quitobench_imageization_protocol.py \
  tests/test_quitobench_proxy_imageization_latency.py \
  -q
```

预期：列出的测试全部通过。

- [ ] **步骤 2：验证输出文件**

运行：

```bash
conda run -n quito python - <<'PY'
import json
from pathlib import Path

import pandas as pd

out_dir = Path("outputs/vision_ts_routing/visual_embeddings/qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e/visual_embedding_cache_smoke_v1")
manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
embeddings = pd.read_parquet(out_dir / "embeddings.parquet")
index = pd.read_csv(out_dir / "embedding_index.csv")
latency = pd.read_csv(out_dir / "latency.csv")
z_cols = [c for c in embeddings.columns if c.startswith("z_")]
assert len(embeddings) == len(index) == manifest["num_windows"]
assert len(z_cols) == manifest["embedding_dim"] == 64
assert embeddings["physical_window_id"].is_unique
assert manifest["view_tensor_semantics"] == "multi_view_not_rgb"
assert manifest["trains_visual_encoder"] is False
assert manifest["runs_expert_models"] is False
assert manifest["implements_router"] is False
assert manifest["reads_expert_errors"] is False
assert manifest["uses_future_target"] is False
assert float(latency["encoder_latency_ms_per_window"].iloc[0]) >= 0.0
print("visual_embedding_smoke_ok", len(embeddings), len(z_cols), manifest["encoder_latency_ms_per_window"])
PY
```

预期输出：

```text
visual_embedding_smoke_ok 288 64 0.0
```

第三个数只要求为非负浮点数，不要求等于 `0.0`。

- [ ] **步骤 3：确认未改动 Stage 1.4g-b 输出**

运行：

```bash
pgrep -af 'quitobench_framework_expert_cache.py|stage14g_b' || true
find outputs/vision_ts_routing/expert_predictions -maxdepth 3 -mmin -30 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
```

这是信息性检查。Stage 1.3a0 不应创建、删除或编辑 `outputs/vision_ts_routing/expert_predictions/` 下的文件。

- [ ] **步骤 4：写实验日志**

创建 `experiment_logs/2026-06-07_2255_stage1_3a0_visual_embedding_cache_smoke.md`：

```markdown
# Stage 1.3a0：Visual Embedding Cache Smoke

## 1. 实验目的

验证 Stage 1.2 `view_tensor [B,3,64,192]` 可以被 visual encoder adapter 消费，并生成与 `physical_window_id` 对齐的 visual embedding cache。本阶段不训练 visual encoder，不实现 router/gate，不读取 expert error，不运行专家模型。

## 2. 实验计划

1. 新增 lightweight visual encoder adapter。
2. 读取 Stage 1.2 smoke tensor 和 image index。
3. 输出 `embeddings.parquet`、`embedding_index.csv`、`latency.csv` 和 `manifest.json`。
4. 验证 embedding 行数、维度、主键唯一性和非目标项标记。
5. 与正在运行的 Stage 1.4g-b expert runner 隔离。

## 3. 执行命令

记录 pytest、CLI smoke、输出校验命令和结果。

## 4. 输入数据与配置

记录 Stage 1.2 输入目录、encoder id、embedding dim、batch size、device。

## 5. 实验结果

记录输出目录、文件列表、embedding shape、latency、manifest 关键字段。

## 6. 问题与观察

记录是否使用 CUDA、是否有依赖问题、是否发现输出路径或并发风险。

## 7. 结论

说明 Stage 1.3a0 是否完成，以及 visual embedding cache 是否可供 Stage 1.5 gate baseline 使用。

## 8. 下一步计划

建议进入 Stage 1.5 前先固定专家池和 oracle target；若需要正式视觉 encoder，应补做原 Stage 1.3a 的三路线比较：`per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding`。
```

- [ ] **步骤 5：更新实验日志总览**

在 `experiment_logs/实验日志总览.md` 追加：

```markdown
| 2026-06-07 22:55 | Stage 1.3a0 | `2026-06-07_2255_stage1_3a0_visual_embedding_cache_smoke.md` | 验证 Stage 1.2 view tensor 可被 visual embedding cache smoke 消费并生成 `physical_window_id` 对齐 embedding cache | 已完成 | 输出 visual embedding smoke cache；embedding 行数与 Stage 1.2 image index 对齐；manifest 确认不训练 encoder、不运行专家、不实现 router；未覆盖三种 adapter 路线比较 | 后续补做原 Stage 1.3a adapter comparison：`per_view_grayscale_repeat`、`learned_1x1_view_adapter`、`custom_patch_embedding` |
```

---

## 自检

- 本计划只覆盖 Stage 1.3a0：adapter smoke、embedding cache、latency、测试、日志。
- 不实现 router/gate。
- 不训练或微调 visual encoder。
- 不读取 expert predictions 或 expert errors。
- 不修改正在运行的 Stage 1.4g-b runner 或 expert output 目录。
- `view_tensor` 始终为 `[B,V,H,W]`；embedding 写为 wide columns：`z_0...z_{D-1}`；join key 始终为 `physical_window_id`。
- smoke encoder 只依赖 `torch`，不引入 `torchvision/timm`。pretrained/frozen ViT、`1x1` view adapter 和 custom patch embedding 比较留到正式 Stage 1.3a。
