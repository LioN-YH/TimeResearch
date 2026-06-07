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
    """Stage 1.3a0 visual embedding cache smoke 配置。"""

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
        if int(view_tensor.shape[1]) != int(self.net[0].in_channels):
            raise ValueError(f"view_tensor view_dim {view_tensor.shape[1]} 与 encoder 输入不一致")
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
    """将 Stage 1.2 view tensor 编码为 smoke embedding。"""

    cfg = config or VisualEncoderSmokeConfig()
    if cfg.batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
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


def build_embedding_table(image_index: pd.DataFrame, embeddings: torch.Tensor, encoder_id: str) -> pd.DataFrame:
    """构造与 `physical_window_id` 对齐的 wide embedding table。"""

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
    missing = set(keep_cols) - set(image_index.columns)
    if missing:
        raise ValueError(f"image_index 缺少列：{sorted(missing)}")
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
    """写出 visual embedding smoke cache。"""

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


def load_stage12_view_tensor(image_tensor_dir: Path) -> tuple[torch.Tensor, pd.DataFrame, dict[str, object]]:
    """读取 Stage 1.2 smoke 输出并校验 tensor/index 对齐。"""

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
    """构造 Stage 1.3a0 manifest。"""

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
