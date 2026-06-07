"""Stage 1.2b：proxy + imageization 在线路径 latency sweep。

本脚本只测量 Stage 1.1b torch proxy kernel 和 Stage 1.2 view tensor
imageization 的在线计算成本。不读取或重算 Stage 1.1 proxy cache，不训练
视觉 encoder，不运行专家模型，不实现 router。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_imageization_protocol import (
    DEFAULT_REGISTRY_DIR,
    ImageizationConfig,
    histories_from_registry,
    imageize_batch,
    load_registry,
    sample_smoke_registry,
)
from tools.quitobench_sample_channel_light_proxy import FEATURE_COLUMNS, compute_light_proxy_torch
from tools.quitobench_window_registry import DEFAULT_DATA_DIR, load_subset_frames


DEFAULT_OUTPUT_DIR = ROOT / "outputs/vision_ts_routing/latency"
DEFAULT_BATCH_SIZES = (1, 8, 32, 128, 512, 1024)
DEFAULT_DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class LatencySweepConfig:
    """Stage 1.2b latency sweep 配置。"""

    stage: str = "stage1_2b_proxy_imageization_latency_sweep"
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    devices: tuple[str, ...] = DEFAULT_DEVICES
    warmup_iters: int = 3
    measure_iters: int = 10
    random_seed: int = 20260607
    image_height: int = 64
    image_width: int = 192


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_component(fn, device: torch.device, warmup_iters: int, measure_iters: int) -> tuple[float, object]:
    """计时单个在线组件，返回平均 batch latency ms 和最后一次输出。"""

    last_output = None
    for _ in range(max(0, warmup_iters)):
        last_output = fn()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(max(1, measure_iters)):
        last_output = fn()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / max(1, measure_iters), last_output


def benchmark_online_components(
    histories: torch.Tensor,
    periods: Sequence[int],
    device: str,
    batch_size: int,
    config: LatencySweepConfig | None = None,
) -> dict[str, object]:
    """测量一个 batch size 下 proxy、view tensor、proxy+view 的在线 latency。

    输入 `histories` 必须已经是 history-only `[N, L]`，函数不会接收 future target，
    也不会读取 Stage 1.1 离线 proxy cache。
    """

    cfg = config or LatencySweepConfig()
    if histories.ndim != 2:
        raise ValueError(f"histories 必须为 [N, L]，当前 shape={tuple(histories.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if len(periods) < batch_size or histories.shape[0] < batch_size:
        raise ValueError("输入样本数必须不少于 batch_size")

    torch_device = torch.device(device)
    batch_histories = histories[:batch_size].to(torch_device)
    batch_periods = torch.as_tensor(periods[:batch_size], dtype=torch.int64, device=torch_device)
    image_config = ImageizationConfig(height=cfg.image_height, width=cfg.image_width)

    def run_proxy() -> torch.Tensor:
        return compute_light_proxy_torch(batch_histories, batch_periods)

    def run_view() -> torch.Tensor:
        view_tensor, _ = imageize_batch(batch_histories, periods=batch_periods.tolist(), config=image_config)
        return view_tensor

    def run_proxy_plus_view() -> tuple[torch.Tensor, torch.Tensor]:
        proxy = compute_light_proxy_torch(batch_histories, batch_periods)
        view_tensor, _ = imageize_batch(batch_histories, periods=batch_periods.tolist(), config=image_config)
        return proxy, view_tensor

    proxy_batch_ms, proxy_output = _time_component(
        run_proxy,
        torch_device,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
    )
    view_batch_ms, view_output = _time_component(
        run_view,
        torch_device,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
    )
    combined_batch_ms, combined_output = _time_component(
        run_proxy_plus_view,
        torch_device,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
    )
    combined_proxy, combined_view = combined_output

    return {
        "stage": cfg.stage,
        "device": torch_device.type,
        "batch_size": int(batch_size),
        "sampled_windows": int(batch_size),
        "history_len": int(batch_histories.shape[1]),
        "warmup_iters": int(cfg.warmup_iters),
        "measure_iters": int(cfg.measure_iters),
        "proxy_batch_latency_ms": float(proxy_batch_ms),
        "view_tensor_batch_latency_ms": float(view_batch_ms),
        "proxy_plus_view_batch_latency_ms": float(combined_batch_ms),
        "proxy_torch_latency_ms_per_window": float(proxy_batch_ms / batch_size),
        "view_tensor_latency_ms_per_window": float(view_batch_ms / batch_size),
        "proxy_plus_view_latency_ms_per_window": float(combined_batch_ms / batch_size),
        "proxy_output_shape": [int(v) for v in proxy_output.shape],
        "view_tensor_shape": [int(v) for v in view_output.shape],
        "combined_proxy_shape": [int(v) for v in combined_proxy.shape],
        "combined_view_tensor_shape": [int(v) for v in combined_view.shape],
        "proxy_feature_count": int(len(FEATURE_COLUMNS)),
        "view_tensor_semantics": "multi_view_not_rgb",
        "runs_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
        "recomputes_stage1_1_cache": False,
    }


def _parse_csv_values(raw: str, cast) -> tuple:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise ValueError(f"空参数：{raw!r}")
    return tuple(values)


def choose_latency_sample(registry: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    """从正式 registry 中抽取 latency 输入样本，尽量覆盖 subset/split/cell。"""

    max_per_group = max(1, sample_size // 48 + 1)
    sampled = sample_smoke_registry(registry, max_per_group=max_per_group, random_seed=random_seed)
    if len(sampled) >= sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_seed)
    elif len(registry) > len(sampled):
        remaining = registry.loc[~registry["physical_window_id"].isin(sampled["physical_window_id"])]
        need = min(sample_size - len(sampled), len(remaining))
        if need > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=need, random_state=random_seed)],
                ignore_index=True,
            )
    return sampled.sort_values(["subset", "item_id", "split", "target_start_idx", "channel"]).reset_index(drop=True)


def run_latency_sweep(
    histories: torch.Tensor,
    periods: Sequence[int],
    config: LatencySweepConfig,
) -> list[dict[str, object]]:
    rows = []
    for device in config.devices:
        if device == "cuda" and not torch.cuda.is_available():
            rows.append(
                {
                    "stage": config.stage,
                    "device": "cuda",
                    "batch_size": None,
                    "sampled_windows": 0,
                    "skipped": True,
                    "skip_reason": "torch.cuda.is_available() is False",
                    "runs_visual_encoder": False,
                    "runs_expert_models": False,
                    "implements_router": False,
                    "recomputes_stage1_1_cache": False,
                }
            )
            continue
        for batch_size in config.batch_sizes:
            if batch_size > histories.shape[0]:
                rows.append(
                    {
                        "stage": config.stage,
                        "device": device,
                        "batch_size": int(batch_size),
                        "sampled_windows": int(histories.shape[0]),
                        "skipped": True,
                        "skip_reason": "sampled_windows < batch_size",
                        "runs_visual_encoder": False,
                        "runs_expert_models": False,
                        "implements_router": False,
                        "recomputes_stage1_1_cache": False,
                    }
                )
                continue
            row = benchmark_online_components(
                histories=histories,
                periods=periods,
                device=device,
                batch_size=int(batch_size),
                config=config,
            )
            row["skipped"] = False
            row["skip_reason"] = ""
            rows.append(row)
    return rows


def build_latency_manifest(
    rows: Sequence[dict[str, object]],
    config: LatencySweepConfig,
    sample_set_id: str,
    base_registry_id: str,
    input_registry_dir: Path,
    sampled_windows: int,
    cuda_available: bool,
) -> dict[str, object]:
    return {
        "stage": config.stage,
        "config": asdict(config),
        "sample_set_id": sample_set_id,
        "base_registry_id": base_registry_id,
        "input_registry_dir": str(input_registry_dir),
        "sampled_windows": int(sampled_windows),
        "rows": int(len(rows)),
        "cuda_available": bool(cuda_available),
        "metrics": [
            "proxy_torch_latency_ms_per_window",
            "view_tensor_latency_ms_per_window",
            "proxy_plus_view_latency_ms_per_window",
        ],
        "feature_columns": FEATURE_COLUMNS,
        "view_names": ["line_raster", "period_fold", "fft_power"],
        "view_tensor_semantics": "multi_view_not_rgb",
        "future_read_policy": "history_only",
        "reads_stage1_1_cache": False,
        "recomputes_stage1_1_cache": False,
        "runs_visual_encoder": False,
        "runs_expert_models": False,
        "implements_router": False,
        "output_files": {
            "latency_csv": "stage1_2b_proxy_imageization_latency.csv",
            "manifest": "stage1_2b_proxy_imageization_latency_manifest.json",
        },
    }


def write_latency_outputs(
    rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    """写出 Stage 1.2b latency CSV 和 manifest。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stage1_2b_proxy_imageization_latency.csv"
    manifest_path = output_dir / "stage1_2b_proxy_imageization_latency_manifest.json"
    pd.DataFrame(list(rows)).to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-sizes", default="1,8,32,128,512,1024")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260607)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_sizes = _parse_csv_values(args.batch_sizes, int)
    devices = _parse_csv_values(args.devices, str)
    config = LatencySweepConfig(
        batch_sizes=batch_sizes,
        devices=devices,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        random_seed=args.random_seed,
    )
    registry, _ = load_registry(args.registry_dir)
    max_batch_size = max(config.batch_sizes)
    sampled = choose_latency_sample(registry, sample_size=max_batch_size, random_seed=config.random_seed)
    subsets = tuple(sorted(sampled["subset"].dropna().unique().tolist()))
    print(f"[input] registry_rows={len(registry)} latency_sample_rows={len(sampled)} subsets={subsets}")
    subset_frames = load_subset_frames(args.data_dir, subsets)
    histories = histories_from_registry(sampled, subset_frames)
    periods = sampled["period"].astype(int).tolist()
    rows = run_latency_sweep(histories=histories, periods=periods, config=config)
    sample_set_values = sorted(sampled["sample_set_id"].dropna().unique().tolist())
    base_registry_values = sorted(sampled["base_registry_id"].dropna().unique().tolist())
    manifest = build_latency_manifest(
        rows=rows,
        config=config,
        sample_set_id=sample_set_values[0] if len(sample_set_values) == 1 else json.dumps(sample_set_values, ensure_ascii=False),
        base_registry_id=base_registry_values[0] if len(base_registry_values) == 1 else json.dumps(base_registry_values, ensure_ascii=False),
        input_registry_dir=args.registry_dir,
        sampled_windows=len(sampled),
        cuda_available=torch.cuda.is_available(),
    )
    csv_path, manifest_path = write_latency_outputs(rows=rows, manifest=manifest, output_dir=args.output_dir)
    print(f"[done] latency_csv={csv_path}")
    print(f"[done] manifest={manifest_path}")
    for row in rows:
        if row.get("skipped"):
            print(f"[skip] device={row.get('device')} batch_size={row.get('batch_size')} reason={row.get('skip_reason')}")
        else:
            print(
                "[metric] "
                f"device={row['device']} batch_size={row['batch_size']} "
                f"proxy={row['proxy_torch_latency_ms_per_window']:.6f}ms/window "
                f"view={row['view_tensor_latency_ms_per_window']:.6f}ms/window "
                f"combined={row['proxy_plus_view_latency_ms_per_window']:.6f}ms/window"
            )


if __name__ == "__main__":
    main()
