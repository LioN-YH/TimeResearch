"""Quito native sanity 工具的轻量纯函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def aggregate_metric_sums(metric_sums: dict[str, float], metrics: Mapping[object, object], batch_size: int) -> None:
    """把 batch mean 指标累加为 sample-weighted sum。"""

    for key, value in metrics.items():
        name = key.name if hasattr(key, "name") else str(key)
        scalar = float(value.item() if hasattr(value, "item") else value)
        metric_sums[name] = metric_sums.get(name, 0.0) + scalar * int(batch_size)


def find_latest_checkpoint(model_task_dir: Path) -> Path:
    """从 Quito finetune 输出目录中选择最新 checkpoint。"""

    candidates = sorted(
        [*model_task_dir.glob("FINE_TUNE/ver_*/checkpoints/*.ckpt"), *model_task_dir.glob("**/checkpoints/*.ckpt")],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到 checkpoint：{model_task_dir}/FINE_TUNE/ver_*/checkpoints/*.ckpt")
    return candidates[-1]


def build_sample_indices(dataset_len: int, stride: int = 1, max_samples: int | None = None) -> list[int]:
    """构造固定 stride 的 dataset index 列表。"""

    if dataset_len < 0:
        raise ValueError("dataset_len 必须非负")
    if stride <= 0:
        raise ValueError("stride 必须为正整数")
    indices = list(range(0, int(dataset_len), int(stride)))
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    return indices


def clear_resume_checkpoint_paths(config) -> None:
    """清空 evaluate YAML 中的 checkpoint，确保 train-eval 从头训练。"""

    if "model" in config and "checkpoint_path" in config.model:
        config.model.checkpoint_path = None
    if "resume" in config and "checkpoint_path" in config.resume:
        config.resume.checkpoint_path = None
