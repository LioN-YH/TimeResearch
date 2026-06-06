"""Stage 1.0：QuitoBench sample-channel 窗口索引与配置注册表。

本脚本只构造窗口级索引，不生成伪图像，不计算 proxy，不运行专家模型。
窗口索引用于后续 Stage 1.1/1.2/1.4 对齐同一个 `window_id`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2023-07-28 00:00:00")
DEFAULT_DATA_DIR = ROOT / "data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315"
DEFAULT_CODEBOOK_PATH = ROOT / "outputs/data_audit/quitobench_official_cluster_codebook.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/vision_ts_routing/window_registry"
SUBSET_PERIODS = {"hour": 24, "min": 144}
DEFAULT_CHANNELS = ("ind_1", "ind_2", "ind_3", "ind_4", "ind_5")


@dataclass(frozen=True)
class RegistryConfig:
    """窗口注册表配置。

    split_strategy 当前实现为 `quito_temporal`：
    - 每个 item 内部先按 QuitoBench cutoff 切 train/valid/test。
    - cutoff 前 80% 为 train，20% 为 valid。
    - cutoff 及之后为 test。
    - 窗口的 history 与 target 都严格落在同一个 split 内。
    """

    history_len: int = 192
    pred_len: int = 96
    stride: int = 96
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    subsets: tuple[str, ...] = ("hour", "min")
    split_strategy: str = "quito_temporal"
    cutoff: str = "2023-07-28 00:00:00"
    item_level_split: bool = False
    dataset: str = "hq-bench/quitobench"
    revision: str = "17362dcb"
    data_version: str = "v20260315"


def _stable_json(data: Mapping[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_config_hash(config: RegistryConfig) -> str:
    """为窗口配置生成稳定短 hash。"""

    return hashlib.sha1(_stable_json(asdict(config)).encode("utf-8")).hexdigest()[:12]


def make_window_id(row_values: Mapping[str, object]) -> str:
    """为单个 sample-channel window 生成稳定 ID。"""

    return hashlib.sha1(_stable_json(row_values).encode("utf-8")).hexdigest()[:16]


def split_bounds_from_dates(dates: pd.Series | pd.DatetimeIndex) -> dict[str, tuple[int, int]]:
    """按 QuitoBench cutoff 重建每个 item 内部 train/valid/test 边界。

    返回值是半开区间 `[start, end)`，下标基于按 `date_time` 排序后的 item 序列。
    """

    dt = pd.to_datetime(pd.Series(dates))
    pre_cutoff_len = int((dt < CUTOFF).sum())
    valid_len = int(pre_cutoff_len * 0.2)
    train_len = pre_cutoff_len - valid_len
    total_len = int(len(dt))
    return {
        "train": (0, train_len),
        "valid": (train_len, pre_cutoff_len),
        "test": (pre_cutoff_len, total_len),
    }


def iter_window_offsets(split_start: int, split_end: int, history_len: int, pred_len: int, stride: int) -> list[int]:
    """生成 target 起点 `start_idx`，保证 history 和 target 均在 split 内。"""

    first_target_start = split_start + history_len
    last_target_start = split_end - pred_len
    if first_target_start > last_target_start:
        return []
    return list(range(first_target_start, last_target_start + 1, stride))


def load_official_codebook(path: Path) -> pd.DataFrame:
    """读取 Stage 0.6b 官方 codebook，并以 cluster code 为 index。"""

    df = pd.read_csv(path)
    required = {"official_cluster_code", "official_tsf_cell"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"官方 codebook 缺少列：{sorted(missing)}")
    return df.set_index("official_cluster_code", drop=False)


def load_subset_frames(data_dir: Path, subsets: tuple[str, ...], max_items_per_subset: int | None = None) -> dict[str, pd.DataFrame]:
    """读取 QuitoBench revision parquet；可用 `max_items_per_subset` 做 smoke。"""

    frames: dict[str, pd.DataFrame] = {}
    for subset in subsets:
        path = data_dir / f"test_{subset}-00001-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少 QuitoBench parquet：{path}")
        df = pd.read_parquet(path)
        df["date_time"] = pd.to_datetime(df["date_time"])
        if max_items_per_subset is not None:
            keep_items = sorted(df["item_id"].dropna().unique())[: int(max_items_per_subset)]
            df = df[df["item_id"].isin(keep_items)].copy()
        frames[subset] = df
    return frames


def build_window_registry(
    subset_frames: Mapping[str, pd.DataFrame],
    codebook: pd.DataFrame,
    config: RegistryConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """构造 sample-channel 窗口索引。

    输出每行对应一个单通道历史窗口。`start_idx` 是 target 起点，
    即 history 结束后的第一个预测位置。
    """

    config_hash = build_config_hash(config)
    if "official_cluster_code" in codebook.columns and codebook.index.name != "official_cluster_code":
        codebook = codebook.set_index("official_cluster_code", drop=False)
    rows: list[dict[str, object]] = []
    subset_item_counts: dict[str, int] = {}
    subset_window_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}

    for subset, df in subset_frames.items():
        if subset not in SUBSET_PERIODS:
            raise ValueError(f"未知 subset：{subset}")
        period = SUBSET_PERIODS[subset]
        indicator_cols = [c for c in config.channels if c in df.columns]
        missing_channels = set(config.channels) - set(indicator_cols)
        if missing_channels:
            raise ValueError(f"{subset} 缺少 channel 列：{sorted(missing_channels)}")
        subset_item_counts[subset] = int(df["item_id"].nunique())
        before_subset = len(rows)

        for item_id, item_df in df.groupby("item_id", sort=True):
            ordered = item_df.sort_values("date_time").reset_index(drop=True)
            cluster_values = ordered["cluster"].dropna().unique()
            if len(cluster_values) != 1:
                raise ValueError(f"{subset}/{item_id} 的 cluster 不唯一：{cluster_values}")
            cluster_code = int(cluster_values[0])
            if cluster_code not in codebook.index:
                raise ValueError(f"cluster code {cluster_code} 不在官方 codebook 中")
            tsf_cell = str(codebook.loc[cluster_code, "official_tsf_cell"])
            bounds = split_bounds_from_dates(ordered["date_time"])

            for split, (split_start, split_end) in bounds.items():
                for target_start in iter_window_offsets(
                    split_start,
                    split_end,
                    config.history_len,
                    config.pred_len,
                    config.stride,
                ):
                    history_start = target_start - config.history_len
                    target_end = target_start + config.pred_len
                    for channel in indicator_cols:
                        base = {
                            "dataset": config.dataset,
                            "revision": config.revision,
                            "data_version": config.data_version,
                            "subset": subset,
                            "item_id": int(item_id),
                            "channel": channel,
                            "split": split,
                            "start_idx": int(target_start),
                            "history_len": int(config.history_len),
                            "pred_len": int(config.pred_len),
                            "stride": int(config.stride),
                            "config_hash": config_hash,
                        }
                        row = {
                            **base,
                            "window_id": make_window_id(base),
                            "period": int(period),
                            "official_cluster": int(cluster_code),
                            "official_tsf_cell": tsf_cell,
                            "history_start_idx": int(history_start),
                            "history_end_idx": int(target_start),
                            "target_start_idx": int(target_start),
                            "target_end_idx": int(target_end),
                            "history_start_time": ordered.loc[history_start, "date_time"].isoformat(),
                            "history_end_time": ordered.loc[target_start - 1, "date_time"].isoformat(),
                            "target_start_time": ordered.loc[target_start, "date_time"].isoformat(),
                            "target_end_time": ordered.loc[target_end - 1, "date_time"].isoformat(),
                            "split_start_idx": int(split_start),
                            "split_end_idx": int(split_end),
                            "item_length": int(len(ordered)),
                            "split_length": int(split_end - split_start),
                        }
                        rows.append(row)
                        split_counts[split] += 1
        subset_window_counts[subset] = len(rows) - before_subset

    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry = registry.sort_values(["subset", "item_id", "split", "start_idx", "channel"]).reset_index(drop=True)
    manifest = {
        "stage": "stage1_0_window_registry",
        "dataset": config.dataset,
        "revision": config.revision,
        "data_version": config.data_version,
        "config_hash": config_hash,
        "config": asdict(config),
        "total_windows": int(len(registry)),
        "subset_item_counts": subset_item_counts,
        "subset_window_counts": subset_window_counts,
        "split_window_counts": split_counts,
        "unique_items": int(registry[["subset", "item_id"]].drop_duplicates().shape[0]) if not registry.empty else 0,
        "unique_channels": sorted(registry["channel"].unique().tolist()) if not registry.empty else [],
    }
    return registry, manifest


def write_simple_yaml(path: Path, data: Mapping[str, object]) -> None:
    """写入无依赖 YAML，避免新增 PyYAML 依赖。"""

    lines = []
    for key, value in data.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_registry_outputs(
    registry: pd.DataFrame,
    manifest: Mapping[str, object],
    config: RegistryConfig,
    output_root: Path,
) -> Path:
    """写出 `window_index.csv`、`config.yml` 和 `manifest.json`。"""

    config_hash = build_config_hash(config)
    out_dir = output_root / config_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_dir / "window_index.csv", index=False)
    write_simple_yaml(out_dir / "config.yml", asdict(config))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--codebook-path", type=Path, default=DEFAULT_CODEBOOK_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--history-len", type=int, default=192)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--subsets", nargs="+", default=["hour", "min"], choices=sorted(SUBSET_PERIODS))
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--max-items-per-subset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegistryConfig(
        history_len=args.history_len,
        pred_len=args.pred_len,
        stride=args.stride,
        channels=tuple(args.channels),
        subsets=tuple(args.subsets),
    )
    print(f"[config] hash={build_config_hash(config)} config={asdict(config)}")
    codebook = load_official_codebook(args.codebook_path)
    frames = load_subset_frames(args.data_dir, config.subsets, max_items_per_subset=args.max_items_per_subset)
    registry, manifest = build_window_registry(frames, codebook, config)
    out_dir = write_registry_outputs(registry, manifest, config, args.output_root)
    print(f"[done] windows={len(registry)} output={out_dir}")
    print(f"[done] split_window_counts={manifest['split_window_counts']}")
    print(f"[done] subset_window_counts={manifest['subset_window_counts']}")


if __name__ == "__main__":
    main()
