"""Quito 原生数据/模型封装的快速 sanity 评估工具。

本脚本读取 Quito 官方 YAML，复用 `AutoConfig`、`load_datasets` 和
`AutoModel`。它不使用 Quito evaluate.py 的 Ray per-user 调度，避免
每个 item 反复 deepcopy 全量 dataset；指标仍来自模型 `eval_step`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))

from quito.config.auto import AutoConfig
from quito.config.training import ModeType, TaskType
from quito.datasets import load_datasets
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.common import set_seed

from tools.quitobench_quito_native_utils import aggregate_metric_sums, build_sample_indices, find_latest_checkpoint


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/quito_native_sanity"


def sample_dataset(dataset, stride: int = 1, max_samples: int | None = None):
    """在 Quito dataset 外层套 Subset，不改变原始 dataset 语义。"""

    indices = build_sample_indices(len(dataset), stride=stride, max_samples=max_samples)
    return Subset(dataset, indices)


def _load_config_with_checkpoint(config_path: Path, checkpoint_path: Path | None):
    config = OmegaConf.load(config_path)
    if checkpoint_path is not None:
        config.resume.checkpoint_path = str(checkpoint_path)
    return config


def evaluate_config(
    config_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    device: str = "cuda:0",
    max_batches: int | None = None,
    eval_batch_size: int | None = None,
    sample_stride: int = 1,
    max_samples: int | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """按 Quito YAML 批量评估 test split，并返回 sample-weighted 指标。"""

    started = time.time()
    config = _load_config_with_checkpoint(config_path, checkpoint_path)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    local_rank = int(device.split(":", 1)[1]) if use_cuda and ":" in device else (-1 if not use_cuda else 0)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        local_rank=local_rank,
        rank=-1,
        world_size=1,
    )
    dataset = load_datasets(
        data_config=data_config,
        task=TaskType.EVALUATE,
        mode=ModeType.TEST,
        cleanup=False,
        concat=True,
    )
    if dataset is None:
        raise ValueError(f"{config_path} 未加载到 test dataset")
    raw_num_samples = len(dataset)
    dataset = sample_dataset(dataset, stride=sample_stride, max_samples=max_samples)

    model = AutoModel.from_config(model_config, local_rank=local_rank)
    if use_cuda:
        model = model.to(device)
        model.device = device
    else:
        model.device = "cpu"
    model.metrics = training_config.eval_metrics
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=int(eval_batch_size or training_config.eval_batch_size),
        shuffle=False,
        num_workers=int(training_config.num_workers),
        pin_memory=bool(training_config.pin_memory and use_cuda),
    )
    metric_sums: dict[str, float] = {}
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            loss_dict, predictions = model.eval_step(batch)
            batch_size = int(len(predictions))
            aggregate_metric_sums(metric_sums, loss_dict, batch_size=batch_size)
            n_samples += batch_size
            n_batches += 1
            if progress_every > 0 and n_batches % int(progress_every) == 0:
                print(f"[progress] eval batches={n_batches} samples={n_samples}", flush=True)
            if max_batches is not None and n_batches >= int(max_batches):
                break
    metrics = {key: value / max(1, n_samples) for key, value in metric_sums.items()}
    result = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "model_name": str(model_config.model_name),
        "seq_len": int(data_config.seq_len),
        "forecast_horizon": int(data_config.forecast_horizon),
        "features": str(data_config.features),
        "num_samples": int(n_samples),
        "raw_num_samples": int(raw_num_samples),
        "sample_stride": int(sample_stride),
        "max_samples": int(max_samples) if max_samples is not None else None,
        "num_batches": int(n_batches),
        "eval_batch_size": int(eval_batch_size or training_config.eval_batch_size),
        "device": device if use_cuda else "cpu",
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{**{k: v for k, v in result.items() if k != "metrics"}, **metrics}]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    return result


def train_then_evaluate_config(
    train_config_path: Path,
    eval_config_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    train_stride: int = 1,
    valid_stride: int = 1,
    test_stride: int = 1,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    max_test_samples: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """使用 Quito trainer 在采样窗口上训练，然后用同口径评估。"""

    started = time.time()
    config = OmegaConf.load(train_config_path)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    local_rank = int(device.split(":", 1)[1]) if use_cuda and ":" in device else (-1 if not use_cuda else 0)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        local_rank=local_rank,
        rank=0 if use_cuda else -1,
        world_size=1 if use_cuda else -1,
    )
    if epochs is not None:
        training_config.num_epochs = int(epochs)
    if batch_size is not None:
        training_config.batch_size = int(batch_size)
    if eval_batch_size is not None:
        training_config.eval_batch_size = int(eval_batch_size)
    if learning_rate is not None:
        training_config.learning_rate = float(learning_rate)
    training_config.output_dir = str(output_dir / "train")
    training_config.num_workers = min(int(training_config.num_workers), 4)
    training_config.logging_steps = int(progress_every) if progress_every > 0 else 0

    set_seed(int(training_config.seed))
    train_dataset = load_datasets(data_config=data_config, task=TaskType.FINE_TUNE, mode=ModeType.TRAIN, concat=True)
    valid_dataset = load_datasets(data_config=data_config, task=TaskType.FINE_TUNE, mode=ModeType.VALID, concat=True)
    if train_dataset is None or valid_dataset is None:
        raise ValueError(f"{train_config_path} 未加载到 train/valid dataset")
    raw_train_samples = len(train_dataset)
    raw_valid_samples = len(valid_dataset)
    train_dataset = sample_dataset(train_dataset, stride=train_stride, max_samples=max_train_samples)
    valid_dataset = sample_dataset(valid_dataset, stride=valid_stride, max_samples=max_valid_samples)

    model = AutoModel.from_config(model_config, local_rank=local_rank)
    if use_cuda:
        model = model.to(device)
        model.device = device
    trainer = AutoTrainer.from_config(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        config=training_config,
        local_rank=local_rank if use_cuda else -1,
        global_rank=0 if use_cuda else -1,
        world_size=1 if use_cuda else -1,
        use_gpu=1 if use_cuda else 0,
    )
    train_result = trainer.train()
    checkpoint_path = find_latest_checkpoint(output_dir / "train")
    eval_result = evaluate_config(
        config_path=eval_config_path,
        output_dir=output_dir / "eval",
        checkpoint_path=checkpoint_path,
        device=device,
        eval_batch_size=eval_batch_size,
        sample_stride=test_stride,
        max_samples=max_test_samples,
        progress_every=progress_every,
    )
    result = {
        "train_config_path": str(train_config_path),
        "eval_config_path": str(eval_config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": eval_result["device"],
        "raw_train_samples": int(raw_train_samples),
        "raw_valid_samples": int(raw_valid_samples),
        "train_samples": int(len(train_dataset)),
        "valid_samples": int(len(valid_dataset)),
        "train_stride": int(train_stride),
        "valid_stride": int(valid_stride),
        "test_stride": int(test_stride),
        "max_train_samples": int(max_train_samples) if max_train_samples is not None else None,
        "max_valid_samples": int(max_valid_samples) if max_valid_samples is not None else None,
        "max_test_samples": int(max_test_samples) if max_test_samples is not None else None,
        "epochs": int(training_config.num_epochs),
        "batch_size": int(training_config.batch_size),
        "eval_batch_size": int(eval_batch_size or training_config.eval_batch_size),
        "train_result": train_result,
        "eval_result": eval_result,
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_eval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "train-eval"), default="evaluate")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--eval-config-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-from-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train-eval":
        if args.eval_config_path is None:
            raise ValueError("--mode train-eval 需要 --eval-config-path")
        result = train_then_evaluate_config(
            train_config_path=args.config_path,
            eval_config_path=args.eval_config_path,
            output_dir=args.output_dir,
            device=args.device,
            train_stride=args.train_stride,
            valid_stride=args.valid_stride,
            test_stride=args.test_stride,
            max_train_samples=args.max_train_samples,
            max_valid_samples=args.max_valid_samples,
            max_test_samples=args.max_test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            progress_every=args.progress_every,
        )
        print(f"[done] output_dir={args.output_dir}")
        print(
            f"[done] checkpoint={result['checkpoint_path']} train_samples={result['train_samples']} "
            f"eval_samples={result['eval_result']['num_samples']} elapsed={result['elapsed_seconds']:.2f}s"
        )
        print("[done] metrics=" + json.dumps(result["eval_result"]["metrics"], ensure_ascii=False, sort_keys=True))
        return

    checkpoint_path = args.checkpoint_path
    if args.checkpoint_from_output is not None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_from_output)
    result = evaluate_config(
        config_path=args.config_path,
        output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
        device=args.device,
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        sample_stride=args.sample_stride,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    print(f"[done] output_dir={args.output_dir}")
    print(f"[done] model={result['model_name']} samples={result['num_samples']} elapsed={result['elapsed_seconds']:.2f}s")
    print("[done] metrics=" + json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
