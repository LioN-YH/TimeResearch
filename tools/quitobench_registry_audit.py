"""QuitoBench registry 分布审计。

本工具只读取 Stage 1 window registry，不读取原始 parquet，不训练模型。
用于检查 canonical sparse registry 的 split/subset/cell/item/channel 覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "physical_window_id",
    "sample_set_id",
    "base_registry_id",
    "split",
    "subset",
    "official_tsf_cell",
    "item_id",
    "channel",
}


def _validate_registry(registry: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"registry 缺少必要列：{sorted(missing)}")
    if registry["physical_window_id"].duplicated().any():
        raise ValueError("registry 中 physical_window_id 不唯一")


def _count_table(registry: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        registry.groupby(group_cols, dropna=False)["physical_window_id"]
        .nunique()
        .reset_index(name="num_windows")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_registry_audit(
    registry: pd.DataFrame,
    manifest: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """构造 registry 分布审计摘要和明细表。"""

    _validate_registry(registry)
    manifest_dict = dict(manifest or {})
    split_subset_cell = _count_table(registry, ["split", "subset", "official_tsf_cell"])
    item_channel = _count_table(registry, ["subset", "item_id", "channel"])
    tables = {
        "split_counts": _count_table(registry, ["split"]),
        "subset_counts": _count_table(registry, ["subset"]),
        "cell_counts": _count_table(registry, ["official_tsf_cell"]),
        "split_subset_cell_counts": split_subset_cell,
        "item_channel_counts": item_channel,
    }
    summary = {
        "total_rows": int(len(registry)),
        "unique_physical_windows": int(registry["physical_window_id"].nunique()),
        "unique_items": int(registry[["subset", "item_id"]].drop_duplicates().shape[0]),
        "unique_channels": int(registry["channel"].nunique()),
        "sample_set_ids": sorted(registry["sample_set_id"].astype(str).unique().tolist()),
        "base_registry_ids": sorted(registry["base_registry_id"].astype(str).unique().tolist()),
        "split_counts": tables["split_counts"].set_index("split")["num_windows"].to_dict(),
        "subset_counts": tables["subset_counts"].set_index("subset")["num_windows"].to_dict(),
        "cell_counts": tables["cell_counts"].set_index("official_tsf_cell")["num_windows"].to_dict(),
        "min_split_subset_cell_windows": int(split_subset_cell["num_windows"].min()) if not split_subset_cell.empty else 0,
        "min_item_channel_windows": int(item_channel["num_windows"].min()) if not item_channel.empty else 0,
        "manifest_total_windows": manifest_dict.get("total_windows"),
        "manifest_sample_set_id": manifest_dict.get("sample_set_id"),
        "manifest_base_registry_id": manifest_dict.get("base_registry_id"),
    }
    return summary, tables


def write_registry_audit_outputs(
    summary: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """写出 audit summary 和分布表。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = args.registry_dir / "window_index.csv"
    manifest_path = args.registry_dir / "manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"缺少 registry：{registry_path}")
    registry = pd.read_csv(registry_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary, tables = build_registry_audit(registry, manifest=manifest)
    output_dir = args.output_dir or (args.registry_dir / "audit")
    out_dir = write_registry_audit_outputs(summary, tables, output_dir)
    print(f"[done] output_dir={out_dir}")
    print(f"[done] total_rows={summary['total_rows']}")
    print(f"[done] split_counts={summary['split_counts']}")
    print(f"[done] min_split_subset_cell_windows={summary['min_split_subset_cell_windows']}")


if __name__ == "__main__":
    main()
