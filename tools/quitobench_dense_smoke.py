"""Quito dense loader smoke.

This wraps the official Quito finetune/eval configs, injects dataset ids for a
small item subset, and reuses the existing native sanity runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))

from quito.config.auto import AutoConfig
from quito.config.training import ModeType, TaskType
from quito.datasets import load_datasets
from tools.quitobench_quito_native_sanity import evaluate_config, train_then_evaluate_config


def _parse_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("item ids 不能为空")
    return ids


def _patch_config(src: Path, dst: Path, hour_ids: list[int], min_ids: list[int]) -> Path:
    config = OmegaConf.load(src)
    datasets = config.data.datasets
    if "TEST_DATA_HOUR" in datasets:
        if hour_ids:
            datasets.TEST_DATA_HOUR.ids = hour_ids
        else:
            del datasets["TEST_DATA_HOUR"]
    if "TEST_DATA_MIN" in datasets:
        if min_ids:
            datasets.TEST_DATA_MIN.ids = min_ids
        else:
            del datasets["TEST_DATA_MIN"]
    if not datasets:
        raise ValueError("至少需要一个 hour 或 min item id")
    dst.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, dst)
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train-eval", "evaluate", "snaive-direct"), default="train-eval")
    parser.add_argument("--train-config-path", type=Path, required=True)
    parser.add_argument("--eval-config-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hour-ids", default="")
    parser.add_argument("--min-ids", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--seasonal-period", type=int, default=6)
    return parser.parse_args()


def evaluate_snaive_direct(
    config_path: Path,
    output_dir: Path,
    hour_ids: list[int],
    min_ids: list[int],
    batch_size: int,
    seasonal_period: int,
    device: str,
) -> dict[str, object]:
    started = time.time()
    patched_eval = _patch_config(config_path, output_dir / "patched_eval.yaml", hour_ids, min_ids)
    config = OmegaConf.load(patched_eval)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    local_rank = int(device.split(":", 1)[1]) if use_cuda and ":" in device else (-1 if not use_cuda else 0)
    data_config, _, training_config = AutoConfig.from_config(
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
        raise ValueError("未加载到 dense test dataset")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_config.num_workers),
        pin_memory=bool(training_config.pin_memory and use_cuda),
    )
    pred_len = int(data_config.forecast_horizon)
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0
    for batch in loader:
        x = batch["x"].to(device if use_cuda else "cpu")
        y = batch["y"][:, -pred_len:, :].to(device if use_cuda else "cpu")
        base = x[:, -int(seasonal_period) :, :]
        repeats = int(math.ceil(pred_len / max(1, base.shape[1])))
        yhat = base.repeat(1, repeats, 1)[:, :pred_len, :]
        diff = yhat - y
        batch_n = int(x.shape[0])
        mse_sum += float(torch.mean(diff * diff).detach().cpu()) * batch_n
        mae_sum += float(torch.mean(torch.abs(diff)).detach().cpu()) * batch_n
        n_samples += batch_n
    result = {
        "config_path": str(config_path),
        "patched_config_path": str(patched_eval),
        "model_name": "seasonal_naive_direct",
        "seasonal_period": int(seasonal_period),
        "hour_ids": hour_ids,
        "min_ids": min_ids,
        "seq_len": int(data_config.seq_len),
        "forecast_horizon": pred_len,
        "features": str(data_config.features),
        "num_samples": int(n_samples),
        "eval_batch_size": int(batch_size),
        "device": device if use_cuda else "cpu",
        "elapsed_seconds": time.time() - started,
        "metrics": {
            "MSE": mse_sum / max(1, n_samples),
            "MAE": mae_sum / max(1, n_samples),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{**{k: v for k, v in result.items() if k != "metrics"}, **result["metrics"]}]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "train-eval" and args.eval_config_path is None:
        raise ValueError("--mode train-eval 需要 --eval-config-path")
    hour_ids = _parse_ids(args.hour_ids) if args.hour_ids else []
    min_ids = _parse_ids(args.min_ids) if args.min_ids else []
    if args.mode == "evaluate":
        patched_eval = _patch_config(args.train_config_path, args.output_dir / "patched_eval.yaml", hour_ids, min_ids)
        result = evaluate_config(
            config_path=patched_eval,
            output_dir=args.output_dir / "eval",
            checkpoint_path=None,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            sample_stride=args.test_stride,
            progress_every=50,
        )
        print("[done] output_dir=" + str(args.output_dir))
        print("[done] eval_metrics=" + str(result["metrics"]))
        return
    if args.mode == "snaive-direct":
        result = evaluate_snaive_direct(
            config_path=args.train_config_path,
            output_dir=args.output_dir,
            hour_ids=hour_ids,
            min_ids=min_ids,
            batch_size=args.eval_batch_size,
            seasonal_period=args.seasonal_period,
            device=args.device,
        )
        print("[done] output_dir=" + str(args.output_dir))
        print("[done] eval_metrics=" + str(result["metrics"]))
        return

    patched_train = _patch_config(args.train_config_path, args.output_dir / "patched_train.yaml", hour_ids, min_ids)
    patched_eval = _patch_config(args.eval_config_path, args.output_dir / "patched_eval.yaml", hour_ids, min_ids)
    result = train_then_evaluate_config(
        train_config_path=patched_train,
        eval_config_path=patched_eval,
        output_dir=args.output_dir,
        device=args.device,
        train_stride=args.train_stride,
        valid_stride=args.valid_stride,
        test_stride=args.test_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        progress_every=50,
    )
    print("[done] output_dir=" + str(args.output_dir))
    print("[done] checkpoint=" + str(result["checkpoint_path"]))
    print("[done] eval_metrics=" + str(result["eval_result"]["metrics"]))


if __name__ == "__main__":
    main()
