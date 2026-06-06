"""Stage 1.2：QuitoBench sample-channel 伪图像协议 smoke。

本脚本只做 history-only view tensor 生成与 smoke 验证，不训练视觉 encoder，
不运行专家模型，不实现 router。正式 imageization 路径使用 torch tensor；
debug PNG 只从已生成 tensor 后处理得到。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_window_registry import DEFAULT_DATA_DIR, load_subset_frames


DEFAULT_SAMPLE_SET_ID = "qb_h192_p96_quito_overlap_8478f330_stride96_2ccfd64e"
DEFAULT_REGISTRY_DIR = ROOT / "outputs/vision_ts_routing/window_registry" / DEFAULT_SAMPLE_SET_ID
DEFAULT_PROXY_DIR = ROOT / "outputs/vision_ts_routing/proxy_features" / DEFAULT_SAMPLE_SET_ID
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/image_tensors"
REQUIRED_REGISTRY_COLUMNS = {
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
    "history_start_idx",
    "history_end_idx",
    "target_start_idx",
    "target_end_idx",
    "history_len",
    "pred_len",
}


@dataclass(frozen=True)
class ImageizationConfig:
    """Stage 1.2 view tensor 协议配置。"""

    stage: str = "stage1_2_imageization_protocol_smoke"
    image_protocol_id: str = "view3_h64_w192_v1"
    height: int = 64
    width: int = 192
    norm_const: float = 0.4
    eps: float = 1e-5
    clip_min: float = -5.0
    clip_max: float = 5.0
    view_names: tuple[str, ...] = ("line_raster", "period_fold", "fft_power")
    max_per_group: int = 8
    random_seed: int = 20260607


def normalize_history_batch(histories: torch.Tensor, config: ImageizationConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """按 single sample-channel history window 独立计算 mean/std 归一化。

    输入 shape 为 `[B, L]`，输出同 shape。这里的 instance 是单个
    `physical_window_id` 的 history，不是全数据集统计。
    """

    if histories.ndim != 2:
        raise ValueError(f"histories 必须为 [B, L]，当前 shape={tuple(histories.shape)}")
    x = histories.to(dtype=torch.float32)
    finite = torch.isfinite(x)
    safe_x = torch.where(finite, x, torch.zeros_like(x))
    counts = finite.sum(dim=1, keepdim=True).clamp_min(1)
    mean = safe_x.sum(dim=1, keepdim=True) / counts
    centered_for_var = torch.where(finite, x - mean, torch.zeros_like(x))
    var = (centered_for_var * centered_for_var).sum(dim=1, keepdim=True) / counts
    std = torch.sqrt(var + config.eps)
    scaled_std = torch.clamp(std / config.norm_const, min=config.eps)
    normalized = torch.where(finite, (x - mean) / scaled_std, torch.zeros_like(x))
    normalized = torch.clamp(normalized, config.clip_min, config.clip_max)
    return normalized, {
        "mean": mean.squeeze(1).detach().cpu(),
        "std": std.squeeze(1).detach().cpu(),
        "norm_const": torch.full((x.shape[0],), float(config.norm_const)),
        "clip_min": torch.full((x.shape[0],), float(config.clip_min)),
        "clip_max": torch.full((x.shape[0],), float(config.clip_max)),
    }


def _scale_clip_to_unit(values: torch.Tensor, config: ImageizationConfig) -> torch.Tensor:
    unit = (values - config.clip_min) / (config.clip_max - config.clip_min)
    return torch.clamp(unit, 0.0, 1.0)


def line_raster_view(normalized: torch.Tensor, config: ImageizationConfig) -> torch.Tensor:
    """生成保留时间轴的 soft point raster view。"""

    bsz, length = normalized.shape
    if length != config.width:
        sampled = F.interpolate(normalized[:, None, :], size=config.width, mode="linear", align_corners=True).squeeze(1)
    else:
        sampled = normalized
    y = _scale_clip_to_unit(sampled, config) * float(config.height - 1)
    rows = torch.arange(config.height, device=normalized.device, dtype=torch.float32).view(1, config.height, 1)
    distance = torch.abs(rows - y[:, None, :])
    raster = torch.clamp(1.0 - distance, min=0.0, max=1.0)
    return raster.reshape(bsz, config.height, config.width)


def period_fold_view(
    normalized: torch.Tensor,
    periods: Sequence[int],
    config: ImageizationConfig,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """按 period 折叠 1D history，再 resize 到固定 `[H, W]`。"""

    views = []
    padding_lengths: list[int] = []
    num_cycles: list[int] = []
    for sample, period in zip(normalized, periods, strict=True):
        period_int = int(period)
        if period_int <= 0:
            raise ValueError(f"period 必须为正整数，当前为 {period}")
        length = int(sample.numel())
        cycles = int(math.ceil(length / period_int))
        padded_len = cycles * period_int
        padding = padded_len - length
        if padding:
            sample = F.pad(sample, (0, padding), value=0.0)
        folded = sample.reshape(cycles, period_int)
        unit = _scale_clip_to_unit(folded, config)
        resized = F.interpolate(
            unit[None, None, :, :],
            size=(config.height, config.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        views.append(resized)
        padding_lengths.append(int(padding))
        num_cycles.append(int(cycles))
    return torch.stack(views, dim=0), padding_lengths, num_cycles


def fft_power_view(normalized: torch.Tensor, config: ImageizationConfig) -> torch.Tensor:
    """生成轻量频域 power view。"""

    centered = normalized - normalized.mean(dim=1, keepdim=True)
    power = torch.abs(torch.fft.rfft(centered, dim=1)) ** 2
    if power.shape[1] > 1:
        power[:, 0] = 0.0
    compressed = torch.log1p(power)
    max_power = compressed.amax(dim=1, keepdim=True).clamp_min(config.eps)
    unit = compressed / max_power
    resized = F.interpolate(
        unit[:, None, None, :],
        size=(config.height, config.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return torch.clamp(resized, 0.0, 1.0)


def imageize_batch(
    histories: torch.Tensor,
    periods: Sequence[int],
    config: ImageizationConfig | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """把 `[B, L]` history batch 转成 `[B, V, H, W]` view tensor。"""

    cfg = config or ImageizationConfig()
    if len(periods) != histories.shape[0]:
        raise ValueError("periods 长度必须等于 batch size")
    normalized, norm_meta = normalize_history_batch(histories, cfg)
    line = line_raster_view(normalized, cfg)
    fold, padding_lengths, num_cycles = period_fold_view(normalized, periods, cfg)
    fft = fft_power_view(normalized, cfg)
    view_tensor = torch.stack([line, fold, fft], dim=1)
    meta: dict[str, object] = {
        "view_names": list(cfg.view_names),
        "view_dim": len(cfg.view_names),
        "view_tensor_semantics": "multi_view_not_rgb",
        "padding_lengths": padding_lengths,
        "num_cycles": num_cycles,
        "normalization": {
            "method": "instance_mean_std",
            "scope": "per_physical_window_id_history",
            "norm_const": cfg.norm_const,
            "eps": cfg.eps,
            "clip_min": cfg.clip_min,
            "clip_max": cfg.clip_max,
            "future_read_policy": "history_only",
        },
        "norm_mean": norm_meta["mean"].numpy().tolist(),
        "norm_std": norm_meta["std"].numpy().tolist(),
    }
    return view_tensor, meta


def validate_registry(registry: pd.DataFrame) -> None:
    missing = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少必要列：{sorted(missing)}")
    if not registry["physical_window_id"].is_unique:
        raise ValueError("registry 中 physical_window_id 不唯一")


def sample_smoke_registry(registry: pd.DataFrame, max_per_group: int, random_seed: int) -> pd.DataFrame:
    """按 subset/split/official_tsf_cell 分层抽样 Stage 1.2 smoke 窗口。"""

    validate_registry(registry)
    if max_per_group <= 0:
        raise ValueError("max_per_group 必须为正整数")
    groups = []
    group_cols = ["subset", "split", "official_tsf_cell"]
    for _, part in registry.groupby(group_cols, sort=True):
        n = min(max_per_group, len(part))
        groups.append(part.sample(n=n, random_state=random_seed))
    sampled = pd.concat(groups, ignore_index=True)
    return sampled.sort_values(group_cols + ["physical_window_id"]).reset_index(drop=True)


def histories_from_registry(sampled_registry: pd.DataFrame, subset_frames: Mapping[str, pd.DataFrame]) -> torch.Tensor:
    """只读取 registry 指定的 history 区间，构造 `[B, L]` tensor。"""

    histories: list[np.ndarray] = []
    frame_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for row in sampled_registry.itertuples(index=False):
        subset = str(row.subset)
        item_id = int(row.item_id)
        channel = str(row.channel)
        key = (subset, item_id)
        if key not in frame_cache:
            item_df = subset_frames[subset][subset_frames[subset]["item_id"] == item_id]
            if item_df.empty:
                raise ValueError(f"原始数据缺少 {subset}/{item_id}")
            frame_cache[key] = item_df.sort_values("date_time").reset_index(drop=True)
        ordered = frame_cache[key]
        if channel not in ordered.columns:
            raise ValueError(f"原始数据缺少 channel：{subset}/{item_id}/{channel}")
        start = int(row.history_start_idx)
        end = int(row.history_end_idx)
        values = ordered[channel].to_numpy(dtype=np.float32)[start:end]
        expected_len = int(row.history_len)
        if len(values) != expected_len:
            raise ValueError(f"{row.physical_window_id} history 长度 {len(values)} != {expected_len}")
        histories.append(values)
    return torch.tensor(np.stack(histories), dtype=torch.float32)


def build_image_index(sampled_registry: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """构造与 view tensor 行顺序一致的 image index。"""

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
        "history_start_idx",
        "history_end_idx",
        "target_start_idx",
        "target_end_idx",
        "history_len",
        "pred_len",
    ]
    image_index = sampled_registry[keep_cols].copy().reset_index(drop=True)
    image_index["norm_mean"] = meta["norm_mean"]
    image_index["norm_std"] = meta["norm_std"]
    image_index["padding_length"] = meta["padding_lengths"]
    image_index["num_cycles"] = meta["num_cycles"]
    image_index["view_tensor_row"] = np.arange(len(image_index), dtype=int)
    return image_index


def save_debug_pngs(view_tensor: torch.Tensor, image_index: pd.DataFrame, debug_dir: Path, debug_png_count: int) -> None:
    """从已生成 view tensor 保存少量三联灰度 debug PNG。"""

    debug_dir.mkdir(parents=True, exist_ok=True)
    count = min(max(0, int(debug_png_count)), int(view_tensor.shape[0]))
    tensor_cpu = view_tensor.detach().cpu().clamp(0.0, 1.0)
    for i in range(count):
        triptych = torch.cat([tensor_cpu[i, j] for j in range(tensor_cpu.shape[1])], dim=1)
        image_array = (triptych.numpy() * 255.0).astype(np.uint8)
        physical_window_id = str(image_index.iloc[i]["physical_window_id"])
        Image.fromarray(image_array, mode="L").save(debug_dir / f"{i:04d}_{physical_window_id}.png")


def write_imageization_outputs(
    view_tensor: torch.Tensor,
    image_index: pd.DataFrame,
    manifest: Mapping[str, object],
    output_root: Path,
    debug_png_count: int = 16,
) -> Path:
    """写出 smoke tensor、index、manifest 和 sampled debug PNG。"""

    sample_set_values = sorted(image_index["sample_set_id"].dropna().unique().tolist())
    if len(sample_set_values) != 1:
        raise ValueError(f"image_index 必须对应唯一 sample_set_id，当前为：{sample_set_values}")
    sample_set_id = str(sample_set_values[0])
    out_dir = output_root / f"{sample_set_id}__stage1_2_smoke_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "view_tensor_sample.npz", view_tensor=view_tensor.detach().cpu().numpy())
    image_index.to_csv(out_dir / "image_index.csv", index=False)
    save_debug_pngs(view_tensor, image_index, out_dir / "debug_png", debug_png_count)
    manifest_to_write = dict(manifest)
    manifest_to_write["output_dir_name"] = out_dir.name
    manifest_to_write["output_files"] = {
        "view_tensor": "view_tensor_sample.npz",
        "image_index": "image_index.csv",
        "manifest": "manifest.json",
        "debug_png_dir": "debug_png",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_to_write, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def load_registry(registry_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    registry_path = registry_dir / "window_index.csv"
    manifest_path = registry_dir / "manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"缺少 registry：{registry_path}")
    registry = pd.read_csv(registry_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return registry, manifest


def validate_proxy_join(proxy_dir: Path, image_index: pd.DataFrame) -> dict[str, object]:
    """验证 Stage 1.2 sample 能按 physical_window_id join Stage 1.1 proxy。"""

    manifest_path = proxy_dir / "manifest.json"
    if not manifest_path.exists():
        return {"proxy_join_checked": False, "proxy_join_rows": 0, "reason": "missing_manifest"}
    proxy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proxy_file = proxy_dir / proxy_manifest["output_file"]
    if proxy_file.suffix == ".parquet":
        proxy_ids = pd.read_parquet(proxy_file, columns=["physical_window_id"])
    else:
        proxy_ids = pd.read_csv(proxy_file, usecols=["physical_window_id"])
    joined = image_index[["physical_window_id"]].merge(proxy_ids, on="physical_window_id", how="inner")
    return {
        "proxy_join_checked": True,
        "proxy_join_rows": int(len(joined)),
        "proxy_total_rows": int(len(proxy_ids)),
        "proxy_manifest": proxy_manifest,
    }


def build_manifest(
    config: ImageizationConfig,
    registry_manifest: Mapping[str, object],
    sampled_registry: pd.DataFrame,
    tensor_meta: Mapping[str, object],
    image_index: pd.DataFrame,
    latency_seconds: float,
    proxy_join: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    sample_set_values = sorted(sampled_registry["sample_set_id"].dropna().unique().tolist())
    base_registry_values = sorted(sampled_registry["base_registry_id"].dropna().unique().tolist())
    return {
        "stage": config.stage,
        "image_protocol_id": config.image_protocol_id,
        "config": asdict(config),
        "sample_set_id": sample_set_values[0] if len(sample_set_values) == 1 else sample_set_values,
        "base_registry_id": base_registry_values[0] if len(base_registry_values) == 1 else base_registry_values,
        "input_registry_dir": str(args.registry_dir),
        "input_proxy_dir": str(args.proxy_dir),
        "input_registry_manifest": registry_manifest,
        "view_names": list(config.view_names),
        "view_dim": len(config.view_names),
        "tensor_shape": [int(len(image_index)), len(config.view_names), config.height, config.width],
        "view_axis_semantics": {
            "line_raster": {"x": "time_index", "y": "normalized_value_height"},
            "period_fold": {"x": "phase_within_period", "y": "cycle_or_period_block"},
            "fft_power": {"x": "frequency_bin", "y": "rasterized_frequency_power"},
        },
        "normalization": tensor_meta["normalization"],
        "sampling": {
            "group_cols": ["subset", "split", "official_tsf_cell"],
            "max_per_group": int(args.max_per_group),
            "random_seed": int(args.random_seed),
            "sampled_windows": int(len(image_index)),
        },
        "channel_policy": "sample_channel_independent",
        "view_tensor_semantics": "multi_view_not_rgb",
        "debug_png_policy": "sampled_only",
        "device": str(args.device),
        "imageization_latency_seconds": float(latency_seconds),
        "imageization_latency_ms_per_window": float(latency_seconds * 1000.0 / max(1, len(image_index))),
        "gpu_latency_measured": str(args.device).startswith("cuda"),
        "proxy_join": dict(proxy_join),
        "unique_physical_window_id": bool(image_index["physical_window_id"].is_unique),
        "runs_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-per-group", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--debug-png-count", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ImageizationConfig(max_per_group=args.max_per_group, random_seed=args.random_seed)
    registry, registry_manifest = load_registry(args.registry_dir)
    sampled = sample_smoke_registry(registry, max_per_group=args.max_per_group, random_seed=args.random_seed)
    subsets = tuple(sorted(sampled["subset"].dropna().unique().tolist()))
    print(f"[input] registry_rows={len(registry)} sampled_rows={len(sampled)} subsets={subsets}")
    frames = load_subset_frames(args.data_dir, subsets)
    histories = histories_from_registry(sampled, frames)
    device = torch.device(args.device)
    histories = histories.to(device)
    periods = sampled["period"].astype(int).tolist()
    started = time.perf_counter()
    view_tensor, tensor_meta = imageize_batch(histories, periods=periods, config=config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_seconds = time.perf_counter() - started
    image_index = build_image_index(sampled, tensor_meta)
    proxy_join = validate_proxy_join(args.proxy_dir, image_index)
    manifest = build_manifest(
        config=config,
        registry_manifest=registry_manifest,
        sampled_registry=sampled,
        tensor_meta=tensor_meta,
        image_index=image_index,
        latency_seconds=latency_seconds,
        proxy_join=proxy_join,
        args=args,
    )
    out_dir = write_imageization_outputs(
        view_tensor=view_tensor,
        image_index=image_index,
        manifest=manifest,
        output_root=args.output_root,
        debug_png_count=args.debug_png_count,
    )
    print(f"[done] output={out_dir}")
    print(f"[done] tensor_shape={manifest['tensor_shape']}")
    print(f"[done] latency_ms_per_window={manifest['imageization_latency_ms_per_window']:.4f}")
    print(f"[done] proxy_join_rows={proxy_join.get('proxy_join_rows')}")


if __name__ == "__main__":
    main()
