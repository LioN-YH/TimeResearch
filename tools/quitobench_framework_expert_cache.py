"""Stage 1.4b：Quito DLinear 训练型专家预测缓存 smoke。

本脚本只接入第一个正式训练型专家，验证训练、推理、`physical_window_id`
映射和 Stage 1.4a 缓存 schema 复用。不实现 router，不运行视觉 encoder，
不修改 Quito 上游代码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
QUITO_ROOT = ROOT / "quito"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if QUITO_ROOT.exists() and str(QUITO_ROOT) not in sys.path:
    sys.path.insert(0, str(QUITO_ROOT))

from quito.config.model import DLinearModelConfig, PatchTSTModelConfig, TSMixerModelConfig
from quito.config.data import DatasetConfig, Features, Freq
from quito.config.training import ModeType
from quito.datasets import TimeSeriesDataset
from quito.models.dlinear import DLinear
from quito.models.patchtst import PatchTST
from quito.models.tsmixer import TSMixer

from tools.quitobench_lightweight_expert_cache import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_DIR,
    build_cell_model_matrix,
    compute_error_table,
    compute_oracle_summary,
    extract_histories_and_targets,
    load_registry,
    validate_registry,
    write_expert_cache_outputs,
)


DLINEAR_EXPERT_ID = "dlinear_quito"
DLINEAR_EXPERT_FAMILY = "decomposition_linear"
PATCHTST_EXPERT_ID = "patchtst_quito"
PATCHTST_EXPERT_FAMILY = "patch_transformer"
TSMIXER_EXPERT_ID = "tsmixer_quito"
TSMIXER_EXPERT_FAMILY = "mlp_mixer"
QUITO_GLOBAL_TEST_POINT = "2023-07-28 00:00:00"
QUITO_SUBSET_FREQ = {"hour": Freq.H, "min": Freq.M}


@dataclass(frozen=True)
class WindowStandardizer:
    """Stage 1.4e wrapper-level train split standardizer."""

    mean: float
    std: float
    scope: str

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - np.float32(self.mean)) / np.float32(self.std)

    def inverse_transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * np.float32(self.std) + np.float32(self.mean)


@dataclass(frozen=True)
class QuitoWindowScaler:
    """Quito TimeSeriesDataset train 段 item/channel scaler。"""

    mean: float
    std: float
    subset: str
    item_id: int
    channel: str
    scope: str = "quito_timeseries_dataset_train_segment"

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - np.float32(self.mean)) / np.float32(self.std)

    def inverse_transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * np.float32(self.std) + np.float32(self.mean)


@dataclass(frozen=True)
class DLinearExpertConfig:
    """Stage 1.4b DLinear smoke 配置。"""

    stage: str = "stage1_4b_dlinear_expert_cache_smoke"
    expert_set_id: str = "dlinear_v1__smoke"
    seq_len: int = 192
    decoder_label_len: int = 0
    pred_len: int = 96
    kernel_size: int = 25
    individual: bool = False
    revin: bool = True
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    train_set_standardize: bool = False
    drop_last: bool = False
    scheduler: str = "none"
    scheduler_t_max: int | None = None
    eta_min: float = 1e-5
    num_workers: int = 0
    eval_batch_size: int | None = None
    random_seed: int = 20260607
    soft_oracle_temperature: float = 1.0


def build_train_split_standardizer(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
) -> WindowStandardizer:
    """用 train split 的 wrapper 窗口值估计全局 mean/std。"""

    train_ids = registry.loc[registry["split"].astype(str) == "train", "physical_window_id"].astype(str).tolist()
    if not train_ids:
        raise ValueError("train-set standardizer 需要至少一个 train window")
    values = [np.asarray(histories[physical_window_id], dtype=np.float32) for physical_window_id in train_ids]
    values.extend(np.asarray(targets[physical_window_id], dtype=np.float32) for physical_window_id in train_ids)
    merged = np.concatenate(values)
    return WindowStandardizer(
        mean=float(np.mean(merged)),
        std=float(np.std(merged) + 1e-8),
        scope="train_split_global_window_values",
    )


def apply_standardizer_to_series_maps(
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    standardizer: WindowStandardizer | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """对 history/target 映射应用同一个 wrapper-level scaler。"""

    if standardizer is None:
        return (
            {key: np.asarray(value, dtype=np.float32) for key, value in histories.items()},
            {key: np.asarray(value, dtype=np.float32) for key, value in targets.items()},
        )
    return (
        {key: standardizer.transform(value) for key, value in histories.items()},
        {key: standardizer.transform(value) for key, value in targets.items()},
    )


def _make_quito_dataset_for_subset(
    subset: str,
    data_dir: Path,
    seq_len: int,
    pred_len: int,
) -> TimeSeriesDataset:
    if subset not in QUITO_SUBSET_FREQ:
        raise ValueError(f"未知 subset：{subset}")
    ds_config = DatasetConfig(
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        file_name=f"test_{subset}-00001-of-00001.parquet",
        is_pretrain=False,
        freq=QUITO_SUBSET_FREQ[subset],
        ds_cls="TimeSeriesDataset",
        target="ind_1",
    )
    return TimeSeriesDataset(
        data_dir=str(data_dir),
        seq_len=seq_len,
        decoder_label_len=0,
        forecast_horizon=pred_len,
        features=Features.S,
        ds_config=ds_config,
        mode=ModeType.TRAIN,
        normalize=True,
        name=f"TEST_DATA_{subset.upper()}",
        cleanup=False,
        global_test_point=QUITO_GLOBAL_TEST_POINT,
    )


def _quito_dataset_lookup(dataset: TimeSeriesDataset) -> tuple[dict[int, int], dict[str, int], dict[int, pd.DataFrame]]:
    if dataset._df is None:
        raise ValueError("Quito TimeSeriesDataset 需要 cleanup=False 以保留原始 dataframe")
    df = dataset._df.copy()
    df[dataset.date_col] = pd.to_datetime(df[dataset.date_col])
    if "item_id" not in df.columns:
        raise ValueError("QuitoBench 数据缺少 item_id，无法映射 registry row")
    df_sorted = df.sort_values(["item_id", dataset.date_col])
    unique_ids = [int(value) for value in df_sorted["item_id"].unique()]
    item_pos = {item_id: idx for idx, item_id in enumerate(unique_ids)}
    channel_pos = {str(channel): idx for idx, channel in enumerate(dataset.feature_cols)}
    item_frames = {
        item_id: group.sort_values(dataset.date_col).reset_index(drop=True)
        for item_id, group in df_sorted.groupby("item_id", sort=True)
    }
    return item_pos, channel_pos, {int(key): value for key, value in item_frames.items()}


def extract_quito_standardized_series_maps(
    registry: pd.DataFrame,
    data_dir: Path,
    progress_every: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, QuitoWindowScaler], dict[str, object]]:
    """复用 Quito TimeSeriesDataset 的 train 段 scaler，并按 registry 抽取窗口。"""

    start = time.perf_counter()
    validate_registry(registry)
    if progress_every > 0:
        print(f"[stage] quito_standardize start rows={len(registry)}", flush=True)
    history_lens = registry["history_len"].astype(int).unique()
    pred_lens = registry["pred_len"].astype(int).unique()
    if len(history_lens) != 1 or len(pred_lens) != 1:
        raise ValueError("Quito 标准化抽取要求单一 history_len/pred_len")
    seq_len = int(history_lens[0])
    pred_len = int(pred_lens[0])
    histories: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    raw_targets: dict[str, np.ndarray] = {}
    scalers: dict[str, QuitoWindowScaler] = {}
    dataset_cache: dict[str, TimeSeriesDataset] = {}
    lookup_cache: dict[str, tuple[dict[int, int], dict[str, int], dict[int, pd.DataFrame]]] = {}

    for row_idx, row in enumerate(registry.itertuples(index=False), start=1):
        subset = str(row.subset)
        if subset not in dataset_cache:
            if progress_every > 0:
                print(f"[stage] quito_standardize build_dataset subset={subset}", flush=True)
            dataset = _make_quito_dataset_for_subset(subset, data_dir=data_dir, seq_len=seq_len, pred_len=pred_len)
            dataset_cache[subset] = dataset
            lookup_cache[subset] = _quito_dataset_lookup(dataset)
            if progress_every > 0:
                print(
                    f"[stage] quito_standardize dataset_ready subset={subset} "
                    f"elapsed={time.perf_counter() - start:.2f}s",
                    flush=True,
                )
        dataset = dataset_cache[subset]
        item_pos, channel_pos, item_frames = lookup_cache[subset]
        physical_window_id = str(row.physical_window_id)
        item_id = int(row.item_id)
        channel = str(row.channel)
        if item_id not in item_pos or item_id not in item_frames:
            raise ValueError(f"Quito dataset 缺少 item：{subset}/{item_id}")
        if channel not in channel_pos:
            raise ValueError(f"Quito dataset 缺少 channel：{subset}/{item_id}/{channel}")
        frame = item_frames[item_id]
        values = frame[channel].to_numpy(dtype=np.float32)
        scaler = QuitoWindowScaler(
            mean=float(dataset.mean[item_pos[item_id], 0, channel_pos[channel]]),
            std=float(dataset.std[item_pos[item_id], 0, channel_pos[channel]]),
            subset=subset,
            item_id=item_id,
            channel=channel,
        )
        history = values[int(row.history_start_idx) : int(row.history_end_idx)]
        target = values[int(row.target_start_idx) : int(row.target_end_idx)]
        if len(history) != int(row.history_len):
            raise ValueError(f"{physical_window_id} history 长度 {len(history)} != {int(row.history_len)}")
        if len(target) != int(row.pred_len):
            raise ValueError(f"{physical_window_id} target 长度 {len(target)} != {int(row.pred_len)}")
        histories[physical_window_id] = scaler.transform(history)
        targets[physical_window_id] = scaler.transform(target)
        raw_targets[physical_window_id] = target.astype(float)
        scalers[physical_window_id] = scaler
        if progress_every > 0 and row_idx % int(progress_every) == 0:
            print(
                "[progress] quito_standardize "
                f"rows={row_idx}/{len(registry)} "
                f"subsets_cached={len(dataset_cache)} "
                f"elapsed={time.perf_counter() - start:.2f}s",
                flush=True,
            )

    summary = {
        "enabled": True,
        "scope": "quito_timeseries_dataset_train_segment",
        "scaler_granularity": "subset_item_channel",
        "source_dataset": "quito.datasets.TimeSeriesDataset",
        "global_test_point": QUITO_GLOBAL_TEST_POINT,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "subsets": sorted(dataset_cache.keys()),
        "num_window_scalers": int(len(scalers)),
    }
    if progress_every > 0:
        print(f"[stage] quito_standardize done elapsed={time.perf_counter() - start:.2f}s", flush=True)
    return histories, targets, raw_targets, scalers, summary


def inverse_transform_prediction_map(
    predictions_by_id: Mapping[str, Sequence[float]],
    scalers_by_id: Mapping[str, QuitoWindowScaler],
) -> dict[str, np.ndarray]:
    """按每个 window 对应的 Quito item/channel scaler 逆变换预测。"""

    restored: dict[str, np.ndarray] = {}
    for physical_window_id, prediction in predictions_by_id.items():
        if physical_window_id not in scalers_by_id:
            raise KeyError(f"缺少 scaler：{physical_window_id}")
        restored[physical_window_id] = scalers_by_id[physical_window_id].inverse_transform(prediction).astype(float)
    return restored


def prepare_model_series_maps(
    registry: pd.DataFrame,
    data_dir: Path,
    train_set_standardize: bool,
    progress_every: int = 0,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, QuitoWindowScaler] | None,
    dict[str, object],
]:
    """准备模型输入尺度和最终误差用 raw target。"""

    if train_set_standardize:
        model_histories, model_targets, raw_targets, scalers, summary = extract_quito_standardized_series_maps(
            registry,
            data_dir=data_dir,
            progress_every=progress_every,
        )
        return model_histories, model_targets, raw_targets, scalers, summary
    raw_histories, raw_targets = extract_histories_and_targets(
        registry,
        data_dir=data_dir,
        progress_every=progress_every,
    )
    model_histories, model_targets = apply_standardizer_to_series_maps(raw_histories, raw_targets, None)
    return model_histories, model_targets, raw_targets, None, {"enabled": False}


@dataclass(frozen=True)
class PatchTSTExpertConfig:
    """Stage 1.4b PatchTST smoke 配置。"""

    stage: str = "stage1_4b_patchtst_expert_cache_smoke"
    expert_set_id: str = "patchtst_v1__stratified_smoke_5k_cuda"
    seq_len: int = 192
    decoder_label_len: int = 0
    pred_len: int = 96
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.05
    fc_dropout: float = 0.05
    head_dropout: float = 0.0
    revin: bool = True
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    train_set_standardize: bool = False
    drop_last: bool = False
    scheduler: str = "none"
    scheduler_t_max: int | None = None
    eta_min: float = 1e-5
    num_workers: int = 0
    eval_batch_size: int | None = None
    random_seed: int = 20260607
    soft_oracle_temperature: float = 1.0


@dataclass(frozen=True)
class TSMixerExpertConfig:
    """Stage 1.4b TSMixer smoke 配置。"""

    stage: str = "stage1_4b_tsmixer_expert_cache_smoke"
    expert_set_id: str = "tsmixer_v1__stratified_smoke_5k_cuda"
    seq_len: int = 192
    decoder_label_len: int = 0
    pred_len: int = 96
    num_blocks: int = 2
    d_ff: int = 64
    norm_type: str = "layer"
    dropout: float = 0.1
    revin: bool = True
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    train_set_standardize: bool = False
    drop_last: bool = False
    scheduler: str = "none"
    scheduler_t_max: int | None = None
    eta_min: float = 1e-5
    num_workers: int = 0
    eval_batch_size: int | None = None
    random_seed: int = 20260607
    soft_oracle_temperature: float = 1.0


class RegistryWindowDataset(Dataset):
    """从 Stage 1.0 registry 映射出的 sample-channel 训练/推理数据集。"""

    def __init__(
        self,
        registry: pd.DataFrame,
        histories: Mapping[str, Sequence[float]],
        targets: Mapping[str, Sequence[float]],
        split: str | None = None,
        require_train_split: bool = False,
    ) -> None:
        validate_registry(registry)
        if require_train_split and split != "train":
            raise ValueError("训练数据只能来自 train split")
        frame = registry.copy()
        if split is not None:
            frame = frame[frame["split"].astype(str) == split].copy()
        self.registry = frame.reset_index(drop=True)
        self.histories = histories
        self.targets = targets
        self.physical_window_ids = self.registry["physical_window_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.registry)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.registry.iloc[idx]
        physical_window_id = str(row["physical_window_id"])
        history = np.asarray(self.histories[physical_window_id], dtype=np.float32)
        target = np.asarray(self.targets[physical_window_id], dtype=np.float32)
        if len(history) != int(row["history_len"]):
            raise ValueError(f"{physical_window_id} history 长度不匹配")
        if len(target) != int(row["pred_len"]):
            raise ValueError(f"{physical_window_id} target 长度不匹配")
        return {
            "physical_window_id": physical_window_id,
            "x": torch.from_numpy(history.reshape(-1, 1)),
            "y": torch.from_numpy(target.reshape(-1, 1)),
        }


def _make_model(config: DLinearExpertConfig, device: str) -> DLinear:
    model_config = DLinearModelConfig(
        model_name="DLinear",
        seq_len=config.seq_len,
        forecast_horizon=config.pred_len,
        decoder_label_len=config.decoder_label_len,
        enc_in=1,
        c_out=1,
        kernel_size=config.kernel_size,
        individual=config.individual,
        revin=config.revin,
    )
    model = DLinear(model_config, local_rank=-1)
    model.setup_loss_fn("mse", {})
    model.to(torch.device(device))
    model.device = device
    return model


def _make_patchtst_model(config: PatchTSTExpertConfig, device: str) -> PatchTST:
    model_config = PatchTSTModelConfig(
        model_name="PatchTST",
        seq_len=config.seq_len,
        forecast_horizon=config.pred_len,
        decoder_label_len=config.decoder_label_len,
        enc_in=1,
        c_out=1,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        dropout=config.dropout,
        fc_dropout=config.fc_dropout,
        head_dropout=config.head_dropout,
        revin=config.revin,
    )
    model = PatchTST(model_config, local_rank=-1)
    model.setup_loss_fn("mse", {})
    model.to(torch.device(device))
    model.device = device
    return model


def _make_tsmixer_model(config: TSMixerExpertConfig, device: str) -> TSMixer:
    model_config = TSMixerModelConfig(
        model_name="TSMixer",
        seq_len=config.seq_len,
        forecast_horizon=config.pred_len,
        decoder_label_len=config.decoder_label_len,
        enc_in=1,
        c_out=1,
        num_blocks=config.num_blocks,
        d_ff=config.d_ff,
        norm_type=config.norm_type,
        dropout=config.dropout,
        revin=config.revin,
    )
    model = TSMixer(model_config, local_rank=-1)
    model.setup_loss_fn("mse", {})
    model.to(torch.device(device))
    model.device = device
    return model


def _train_model(
    model: DLinear | PatchTST | TSMixer,
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    drop_last: bool,
    scheduler: str,
    scheduler_t_max: int | None,
    eta_min: float,
    num_workers: int,
    random_seed: int,
    device: str,
    progress_every: int = 0,
) -> dict[str, object]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    train_dataset = RegistryWindowDataset(registry, histories, targets, split="train", require_train_split=True)
    if len(train_dataset) == 0:
        raise ValueError("训练型专家 smoke 缺少 train split 窗口")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_t_max) if scheduler_t_max is not None else max(epochs, 1),
            eta_min=eta_min,
        )
    elif scheduler != "none":
        raise ValueError(f"不支持的 scheduler：{scheduler}")

    losses: list[float] = []
    start = time.perf_counter()
    if progress_every > 0:
        print(
            f"[stage] train start windows={len(train_dataset)} batches_per_epoch={len(loader)} "
            f"epochs={epochs} device={device}",
            flush=True,
        )
    global_step = 0
    for epoch_idx in range(epochs):
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x=x, y=y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if progress_every > 0 and global_step % int(progress_every) == 0:
                print(
                    "[progress] train "
                    f"epoch={epoch_idx + 1}/{epochs} batch={batch_idx}/{len(loader)} "
                    f"step={global_step} loss={losses[-1]:.6g} "
                    f"elapsed={time.perf_counter() - start:.2f}s",
                    flush=True,
                )
        if lr_scheduler is not None:
            lr_scheduler.step()
    if progress_every > 0:
        print(f"[stage] train done elapsed={time.perf_counter() - start:.2f}s", flush=True)
    elapsed = time.perf_counter() - start
    return {
        "train_windows": int(len(train_dataset)),
        "trained_splits": ["train"],
        "epochs_completed": int(epochs),
        "final_train_loss": float(losses[-1]) if losses else np.nan,
        "train_elapsed_seconds": float(elapsed),
        "drop_last": bool(drop_last),
        "scheduler": scheduler,
        "scheduler_t_max": int(scheduler_t_max) if scheduler_t_max is not None else max(int(epochs), 1),
        "eta_min": float(eta_min),
        "num_workers": int(num_workers),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def train_quito_dlinear_model(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    config: DLinearExpertConfig | None = None,
    device: str = "cpu",
    progress_every: int = 0,
) -> tuple[DLinear, dict[str, object]]:
    """只使用 train split 训练 Quito DLinear。"""

    cfg = config or DLinearExpertConfig()
    model = _make_model(cfg, device=device)
    stats = _train_model(
        model,
        registry,
        histories,
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        drop_last=cfg.drop_last,
        scheduler=cfg.scheduler,
        scheduler_t_max=cfg.scheduler_t_max,
        eta_min=cfg.eta_min,
        num_workers=cfg.num_workers,
        random_seed=cfg.random_seed,
        device=device,
        progress_every=progress_every,
    )
    return model, stats


def train_quito_patchtst_model(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    config: PatchTSTExpertConfig | None = None,
    device: str = "cpu",
    progress_every: int = 0,
) -> tuple[PatchTST, dict[str, object]]:
    """只使用 train split 训练 Quito PatchTST。"""

    cfg = config or PatchTSTExpertConfig()
    model = _make_patchtst_model(cfg, device=device)
    stats = _train_model(
        model,
        registry,
        histories,
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        drop_last=cfg.drop_last,
        scheduler=cfg.scheduler,
        scheduler_t_max=cfg.scheduler_t_max,
        eta_min=cfg.eta_min,
        num_workers=cfg.num_workers,
        random_seed=cfg.random_seed,
        device=device,
        progress_every=progress_every,
    )
    return model, stats


def train_quito_tsmixer_model(
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    config: TSMixerExpertConfig | None = None,
    device: str = "cpu",
    progress_every: int = 0,
) -> tuple[TSMixer, dict[str, object]]:
    """只使用 train split 训练 Quito TSMixer。"""

    cfg = config or TSMixerExpertConfig()
    model = _make_tsmixer_model(cfg, device=device)
    stats = _train_model(
        model,
        registry,
        histories,
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        drop_last=cfg.drop_last,
        scheduler=cfg.scheduler,
        scheduler_t_max=cfg.scheduler_t_max,
        eta_min=cfg.eta_min,
        num_workers=cfg.num_workers,
        random_seed=cfg.random_seed,
        device=device,
        progress_every=progress_every,
    )
    return model, stats


def predict_with_model(
    model: DLinear | PatchTST | TSMixer,
    registry: pd.DataFrame,
    histories: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    config: DLinearExpertConfig | PatchTSTExpertConfig | TSMixerExpertConfig | None = None,
    device: str = "cpu",
    output_standardizer: WindowStandardizer | None = None,
    progress_every: int = 0,
) -> dict[str, np.ndarray]:
    cfg = config or DLinearExpertConfig()
    dataset = RegistryWindowDataset(registry, histories, targets)
    batch_size = cfg.eval_batch_size or cfg.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    predictions: dict[str, np.ndarray] = {}
    model.eval()
    start = time.perf_counter()
    if progress_every > 0:
        print(f"[stage] predict start windows={len(dataset)} batches={len(loader)} device={device}", flush=True)
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch_count += 1
            x = batch["x"].to(device)
            yhat = model.predict(x=x, y=None).detach().cpu().numpy()
            ids = [str(value) for value in batch["physical_window_id"]]
            for physical_window_id, pred in zip(ids, yhat, strict=True):
                values = pred[:, 0].astype(np.float32)
                if output_standardizer is not None:
                    values = output_standardizer.inverse_transform(values)
                predictions[physical_window_id] = values.astype(float)
            if progress_every > 0 and batch_count % int(progress_every) == 0:
                print(
                    f"[progress] predict batches={batch_count}/{len(loader)} "
                    f"predictions={len(predictions)} elapsed={time.perf_counter() - start:.2f}s",
                    flush=True,
                )
    if progress_every > 0:
        print(f"[stage] predict done elapsed={time.perf_counter() - start:.2f}s", flush=True)
    return predictions


def build_dlinear_prediction_table(
    registry: pd.DataFrame,
    predictions_by_id: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """把 DLinear 输出转换为 Stage 1.4a 兼容的 wide prediction table。"""

    return _build_single_expert_prediction_table(registry, predictions_by_id, DLINEAR_EXPERT_ID, DLINEAR_EXPERT_FAMILY)


def _build_single_expert_prediction_table(
    registry: pd.DataFrame,
    predictions_by_id: Mapping[str, Sequence[float]],
    expert_id: str,
    expert_family: str,
) -> pd.DataFrame:
    validate_registry(registry)
    rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        physical_window_id = str(row.physical_window_id)
        if physical_window_id not in predictions_by_id:
            raise KeyError(f"缺少 prediction：{physical_window_id}")
        prediction = np.asarray(predictions_by_id[physical_window_id], dtype=float)
        pred_len = int(row.pred_len)
        if len(prediction) != pred_len:
            raise ValueError(f"{physical_window_id} prediction 长度 {len(prediction)} != {pred_len}")
        out_row = {
            "physical_window_id": physical_window_id,
            "window_id": str(row.window_id),
            "base_registry_id": str(row.base_registry_id),
            "sample_set_id": str(row.sample_set_id),
            "subset": str(row.subset),
            "split": str(row.split),
            "item_id": str(row.item_id),
            "channel": str(row.channel),
            "period": int(row.period),
            "official_tsf_cell": str(row.official_tsf_cell),
            "history_start_idx": int(row.history_start_idx),
            "target_start_idx": int(row.target_start_idx),
            "pred_len": pred_len,
            "expert_id": expert_id,
            "expert_family": expert_family,
            "prediction_format": "wide_columns",
        }
        for horizon_idx, value in enumerate(prediction):
            out_row[f"yhat_{horizon_idx}"] = float(value)
        rows.append(out_row)
    predictions = pd.DataFrame(rows)
    if predictions[["physical_window_id", "expert_id"]].duplicated().any():
        raise ValueError("predictions 中 (physical_window_id, expert_id) 不唯一")
    return predictions


def build_patchtst_prediction_table(
    registry: pd.DataFrame,
    predictions_by_id: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """把 PatchTST 输出转换为 Stage 1.4a 兼容的 wide prediction table。"""

    return _build_single_expert_prediction_table(registry, predictions_by_id, PATCHTST_EXPERT_ID, PATCHTST_EXPERT_FAMILY)


def build_tsmixer_prediction_table(
    registry: pd.DataFrame,
    predictions_by_id: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """把 TSMixer 输出转换为 Stage 1.4a 兼容的 wide prediction table。"""

    return _build_single_expert_prediction_table(registry, predictions_by_id, TSMIXER_EXPERT_ID, TSMIXER_EXPERT_FAMILY)


def audit_framework_reuse() -> dict[str, object]:
    return {
        "quito_root": str(QUITO_ROOT),
        "quito_has_dlinear": (QUITO_ROOT / "quito/models/dlinear.py").exists(),
        "quito_has_timeseries_dataset": (QUITO_ROOT / "quito/datasets.py").exists(),
        "time_series_library_root": str(ROOT.parents[2] / "Time-Series-Library"),
        "time_series_library_available": (ROOT.parents[2] / "Time-Series-Library/models/DLinear.py").exists(),
        "tslib_available": False,
        "selected_smoke_path": "Quito DLinear model + project registry thin wrapper",
    }


def select_stratified_registry(
    registry: pd.DataFrame,
    max_rows: int,
    group_cols: Sequence[str] = ("split", "subset", "official_tsf_cell"),
    random_seed: int = 20260607,
) -> pd.DataFrame:
    """按 split/subset/cell 等字段做确定性分层抽样。"""

    validate_registry(registry)
    if max_rows <= 0:
        raise ValueError("max_rows 必须为正整数")
    missing = set(group_cols) - set(registry.columns)
    if missing:
        raise ValueError(f"分层抽样缺少列：{sorted(missing)}")
    if max_rows >= len(registry):
        return registry.copy().reset_index(drop=True)

    grouped = list(registry.groupby(list(group_cols), sort=True, dropna=False))
    quota = max(1, max_rows // max(len(grouped), 1))
    remainder = max_rows - quota * len(grouped)
    samples: list[pd.DataFrame] = []
    used_indices: set[int] = set()
    for group_idx, (_, group) in enumerate(grouped):
        take = quota + (1 if group_idx < remainder else 0)
        take = min(take, len(group))
        sample = group.sample(n=take, random_state=random_seed + group_idx)
        samples.append(sample)
        used_indices.update(int(idx) for idx in sample.index)

    sampled = pd.concat(samples, ignore_index=False) if samples else registry.iloc[[]]
    if len(sampled) < max_rows:
        remaining = registry.drop(index=list(used_indices), errors="ignore")
        fill = remaining.sample(n=min(max_rows - len(sampled), len(remaining)), random_state=random_seed + 100_003)
        sampled = pd.concat([sampled, fill], ignore_index=False)
    return sampled.sort_values(["split", "subset", "official_tsf_cell", "physical_window_id"]).head(max_rows).reset_index(drop=True)


def build_dlinear_cache_manifest(
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    elapsed_seconds: float,
    input_registry_dir: Path,
    max_rows: int | None,
    config: DLinearExpertConfig | None = None,
    training_stats: Mapping[str, object] | None = None,
    audit_summary: Mapping[str, object] | None = None,
    sampling_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cfg = config or DLinearExpertConfig()
    return {
        "stage": cfg.stage,
        "expert_set_id": cfg.expert_set_id,
        "expert_ids": [DLINEAR_EXPERT_ID],
        "expert_families": {DLINEAR_EXPERT_ID: DLINEAR_EXPERT_FAMILY},
        "source_framework": "quito",
        "source_model": "quito.models.dlinear.DLinear",
        "sample_set_id": sorted(registry["sample_set_id"].astype(str).unique().tolist()),
        "base_registry_id": sorted(registry["base_registry_id"].astype(str).unique().tolist()),
        "input_registry_dir": str(input_registry_dir),
        "max_rows": max_rows,
        "total_windows": int(registry["physical_window_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "error_rows": int(len(errors)),
        "unique_prediction_key": bool(not predictions[["physical_window_id", "expert_id"]].duplicated().any()),
        "unique_error_key": bool(not errors[["physical_window_id", "expert_id"]].duplicated().any()),
        "split_window_counts": registry.groupby("split")["physical_window_id"].nunique().to_dict(),
        "prediction_split_counts": predictions.groupby("split")["physical_window_id"].nunique().to_dict(),
        "prediction_format": "wide_columns",
        "future_read_policy": "history_only_for_prediction",
        "target_usage": "loss_error_and_oracle_only",
        "implements_router": False,
        "runs_visual_encoder": False,
        "runs_neural_experts": True,
        "modifies_quito_code": False,
        "elapsed_seconds": float(elapsed_seconds),
        "latency_ms_per_window": float(elapsed_seconds * 1000.0 / max(len(registry), 1)),
        "config": asdict(cfg),
        "training_stats": dict(training_stats or {}),
        "audit_summary": dict(audit_summary or audit_framework_reuse()),
        "sampling_summary": dict(sampling_summary or {"strategy": "head" if max_rows is not None else "full_or_external"}),
        "output_files": {
            "predictions": "predictions.parquet",
            "errors": "errors.parquet",
            "manifest": "manifest.json",
            "cell_model_matrix": "profiling/cell_model_matrix.csv",
            "oracle_summary": "profiling/oracle_summary.csv",
        },
    }


def build_patchtst_cache_manifest(
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    elapsed_seconds: float,
    input_registry_dir: Path,
    max_rows: int | None,
    config: PatchTSTExpertConfig | None = None,
    training_stats: Mapping[str, object] | None = None,
    audit_summary: Mapping[str, object] | None = None,
    sampling_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cfg = config or PatchTSTExpertConfig()
    manifest = build_dlinear_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=elapsed_seconds,
        input_registry_dir=input_registry_dir,
        max_rows=max_rows,
        config=DLinearExpertConfig(stage=cfg.stage, expert_set_id=cfg.expert_set_id),
        training_stats=training_stats,
        audit_summary=audit_summary,
        sampling_summary=sampling_summary,
    )
    manifest["stage"] = cfg.stage
    manifest["expert_set_id"] = cfg.expert_set_id
    manifest["expert_ids"] = [PATCHTST_EXPERT_ID]
    manifest["expert_families"] = {PATCHTST_EXPERT_ID: PATCHTST_EXPERT_FAMILY}
    manifest["source_model"] = "quito.models.patchtst.PatchTST"
    manifest["config"] = asdict(cfg)
    return manifest


def build_tsmixer_cache_manifest(
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    elapsed_seconds: float,
    input_registry_dir: Path,
    max_rows: int | None,
    config: TSMixerExpertConfig | None = None,
    training_stats: Mapping[str, object] | None = None,
    audit_summary: Mapping[str, object] | None = None,
    sampling_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cfg = config or TSMixerExpertConfig()
    manifest = build_dlinear_cache_manifest(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=elapsed_seconds,
        input_registry_dir=input_registry_dir,
        max_rows=max_rows,
        config=DLinearExpertConfig(stage=cfg.stage, expert_set_id=cfg.expert_set_id),
        training_stats=training_stats,
        audit_summary=audit_summary,
        sampling_summary=sampling_summary,
    )
    manifest["stage"] = cfg.stage
    manifest["expert_set_id"] = cfg.expert_set_id
    manifest["expert_ids"] = [TSMIXER_EXPERT_ID]
    manifest["expert_families"] = {TSMIXER_EXPERT_ID: TSMIXER_EXPERT_FAMILY}
    manifest["source_model"] = "quito.models.tsmixer.TSMixer"
    manifest["config"] = asdict(cfg)
    return manifest


def _select_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _infer_window_lengths(registry: pd.DataFrame) -> tuple[int, int]:
    history_lens = sorted(registry["history_len"].astype(int).unique().tolist())
    pred_lens = sorted(registry["pred_len"].astype(int).unique().tolist())
    if len(history_lens) != 1 or len(pred_lens) != 1:
        raise ValueError(f"registry 必须只有一种 history_len/pred_len：{history_lens}/{pred_lens}")
    return int(history_lens[0]), int(pred_lens[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-model", choices=("dlinear", "patchtst", "tsmixer"), default="dlinear")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--expert-set-id", default="dlinear_v1__smoke")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--stratified-rows", type=int, default=None)
    parser.add_argument("--stratify-cols", default="split,subset,official_tsf_cell")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-set-standardize", action="store_true", default=False)
    parser.add_argument("--drop-last", action="store_true", default=False)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--decoder-label-len", type=int, default=0)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=25)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--norm-type", default="layer")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fc-dropout", type=float, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--revin", dest="revin", action="store_true", default=True)
    parser.add_argument("--no-revin", dest="revin", action="store_false")
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expert_model == "patchtst":
        config = PatchTSTExpertConfig(
            expert_set_id=args.expert_set_id,
            epochs=args.epochs,
            decoder_label_len=args.decoder_label_len,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_set_standardize=args.train_set_standardize,
            drop_last=args.drop_last,
            scheduler=args.scheduler,
            scheduler_t_max=args.scheduler_t_max,
            eta_min=args.eta_min,
            num_workers=args.num_workers,
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
            random_seed=args.random_seed,
        )
    elif args.expert_model == "tsmixer":
        config = TSMixerExpertConfig(
            expert_set_id=args.expert_set_id,
            epochs=args.epochs,
            decoder_label_len=args.decoder_label_len,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_set_standardize=args.train_set_standardize,
            drop_last=args.drop_last,
            scheduler=args.scheduler,
            scheduler_t_max=args.scheduler_t_max,
            eta_min=args.eta_min,
            num_workers=args.num_workers,
            num_blocks=args.num_blocks,
            d_ff=args.d_ff,
            norm_type=args.norm_type,
            dropout=args.dropout if args.dropout is not None else TSMixerExpertConfig.dropout,
            revin=args.revin,
            random_seed=args.random_seed,
        )
    else:
        config = DLinearExpertConfig(
            expert_set_id=args.expert_set_id,
            epochs=args.epochs,
            decoder_label_len=args.decoder_label_len,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_set_standardize=args.train_set_standardize,
            drop_last=args.drop_last,
            scheduler=args.scheduler,
            scheduler_t_max=args.scheduler_t_max,
            eta_min=args.eta_min,
            num_workers=args.num_workers,
            kernel_size=args.kernel_size,
            revin=args.revin,
            random_seed=args.random_seed,
        )
    start = time.perf_counter()
    load_max_rows = None if args.stratified_rows is not None else args.max_rows
    if args.progress_every > 0:
        print(f"[stage] load_registry start dir={args.registry_dir}", flush=True)
    registry, registry_manifest = load_registry(args.registry_dir, max_rows=load_max_rows)
    if args.progress_every > 0:
        print(f"[stage] load_registry done rows={len(registry)} elapsed={time.perf_counter() - start:.2f}s", flush=True)
    sampling_summary: dict[str, object]
    if args.stratified_rows is not None:
        stratify_cols = tuple(col.strip() for col in args.stratify_cols.split(",") if col.strip())
        registry = select_stratified_registry(
            registry,
            max_rows=args.stratified_rows,
            group_cols=stratify_cols,
            random_seed=config.random_seed,
        )
        sampling_summary = {
            "strategy": "stratified",
            "requested_rows": int(args.stratified_rows),
            "selected_rows": int(len(registry)),
            "group_cols": list(stratify_cols),
            "group_count": int(registry.groupby(list(stratify_cols))["physical_window_id"].nunique().shape[0]),
        }
        if args.progress_every > 0:
            print(
                f"[stage] stratified_sample done rows={len(registry)} "
                f"elapsed={time.perf_counter() - start:.2f}s",
                flush=True,
            )
    else:
        sampling_summary = {
            "strategy": "head" if args.max_rows is not None else "full",
            "requested_rows": args.max_rows,
            "selected_rows": int(len(registry)),
        }
    if args.max_train_windows is not None:
        train = registry[registry["split"].astype(str) == "train"].head(args.max_train_windows)
        other = registry[registry["split"].astype(str) != "train"]
        registry = pd.concat([train, other], ignore_index=True)
        validate_registry(registry)
        sampling_summary["max_train_windows"] = int(args.max_train_windows)
        sampling_summary["selected_rows_after_train_cap"] = int(len(registry))
    inferred_seq_len, inferred_pred_len = _infer_window_lengths(registry)
    config = replace(
        config,
        seq_len=int(args.seq_len or inferred_seq_len),
        pred_len=int(args.pred_len or inferred_pred_len),
    )
    if config.seq_len != inferred_seq_len or config.pred_len != inferred_pred_len:
        raise ValueError(
            "模型窗口长度必须匹配 registry："
            f"config=({config.seq_len},{config.pred_len}) registry=({inferred_seq_len},{inferred_pred_len})"
        )
    if args.progress_every > 0:
        print(f"[stage] model_window_lengths seq_len={config.seq_len} pred_len={config.pred_len}", flush=True)
    sample_set_ids = sorted(registry["sample_set_id"].astype(str).unique().tolist())
    if len(sample_set_ids) != 1:
        raise ValueError(f"单次 expert cache 只支持一个 sample_set_id：{sample_set_ids}")
    output_root = args.output_root or (DEFAULT_OUTPUT_ROOT.parent / sample_set_ids[0])
    model_histories, model_targets, targets, prediction_scalers, standardization_summary = prepare_model_series_maps(
        registry,
        data_dir=args.data_dir,
        train_set_standardize=config.train_set_standardize,
        progress_every=args.progress_every,
    )
    if args.progress_every > 0:
        print(f"[stage] prepare_model_series done elapsed={time.perf_counter() - start:.2f}s", flush=True)
    device = _select_device(args.device)
    if args.expert_model == "patchtst":
        model, training_stats = train_quito_patchtst_model(
            registry,
            model_histories,
            model_targets,
            config=config,
            device=device,
            progress_every=args.progress_every,
        )
    elif args.expert_model == "tsmixer":
        model, training_stats = train_quito_tsmixer_model(
            registry,
            model_histories,
            model_targets,
            config=config,
            device=device,
            progress_every=args.progress_every,
        )
    else:
        model, training_stats = train_quito_dlinear_model(
            registry,
            model_histories,
            model_targets,
            config=config,
            device=device,
            progress_every=args.progress_every,
        )
    prediction_map = predict_with_model(
        model,
        registry,
        model_histories,
        model_targets,
        config=config,
        device=device,
        progress_every=args.progress_every,
    )
    if args.progress_every > 0:
        print(f"[stage] build_outputs start elapsed={time.perf_counter() - start:.2f}s", flush=True)
    if prediction_scalers is not None:
        prediction_map = inverse_transform_prediction_map(prediction_map, prediction_scalers)
    predictions = (
        build_patchtst_prediction_table(registry, prediction_map)
        if args.expert_model == "patchtst"
        else build_tsmixer_prediction_table(registry, prediction_map)
        if args.expert_model == "tsmixer"
        else build_dlinear_prediction_table(registry, prediction_map)
    )
    errors = compute_error_table(predictions, targets)
    oracle_summary = compute_oracle_summary(errors)
    cell_model_matrix = build_cell_model_matrix(errors)
    elapsed = time.perf_counter() - start
    if args.expert_model == "patchtst":
        manifest_builder = build_patchtst_cache_manifest
    elif args.expert_model == "tsmixer":
        manifest_builder = build_tsmixer_cache_manifest
    else:
        manifest_builder = build_dlinear_cache_manifest
    manifest = manifest_builder(
        registry=registry,
        predictions=predictions,
        errors=errors,
        elapsed_seconds=elapsed,
        input_registry_dir=args.registry_dir,
        max_rows=args.max_rows,
        config=config,
        training_stats={**training_stats, "device": device},
        audit_summary={**audit_framework_reuse(), "selected_expert_model": args.expert_model},
        sampling_summary=sampling_summary,
    )
    manifest["input_registry_manifest"] = registry_manifest
    manifest["standardization"] = standardization_summary
    out_dir = write_expert_cache_outputs(
        predictions=predictions,
        errors=errors,
        oracle_summary=oracle_summary,
        cell_model_matrix=cell_model_matrix,
        manifest=manifest,
        output_root=output_root,
        expert_set_id=args.expert_set_id,
    )
    print(f"[done] output_dir={out_dir}")
    print(f"[done] windows={manifest['total_windows']}")
    print(f"[done] prediction_rows={manifest['prediction_rows']}")
    print(f"[done] train_windows={manifest['training_stats']['train_windows']}")


if __name__ == "__main__":
    main()
