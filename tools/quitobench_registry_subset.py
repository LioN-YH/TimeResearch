"""Materialize a fixed sampled QuitoBench window registry.

The output is a first-class registry directory: every downstream expert should
read its `window_index.csv` directly and avoid additional sampling.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quitobench_common import load_json_manifest, require_columns, write_json_manifest


REQUIRED_COLUMNS = {
    "physical_window_id",
    "window_id",
    "base_registry_id",
    "sample_set_id",
    "split",
    "subset",
    "item_id",
    "channel",
    "official_tsf_cell",
}


def _parse_stratify_cols(value: str) -> list[str]:
    cols = [part.strip() for part in value.split(",") if part.strip()]
    if not cols:
        raise ValueError("--stratify-cols 至少需要一列")
    return cols


def _allocate_stratified_counts(group_sizes: pd.Series, target_rows: int) -> pd.Series:
    if target_rows <= 0:
        raise ValueError("target_rows 必须为正数")
    total_rows = int(group_sizes.sum())
    if target_rows > total_rows:
        raise ValueError(f"target_rows={target_rows} 大于输入行数 {total_rows}")
    if target_rows == total_rows:
        return group_sizes.astype(int)

    exact = group_sizes.astype(float) * float(target_rows) / float(total_rows)
    counts = exact.apply(int)
    remainder = int(target_rows - counts.sum())
    if remainder:
        order = (
            pd.DataFrame(
                {
                    "fraction": exact - counts,
                    "size": group_sizes,
                }
            )
            .sort_values(["fraction", "size"], ascending=[False, False])
            .index
        )
        for key in order[:remainder]:
            counts.loc[key] += 1
    if (counts > group_sizes).any():
        raise ValueError("内部错误：分层抽样数量超过 group size")
    return counts.astype(int)


def _sample_registry(
    registry: pd.DataFrame,
    target_rows: int,
    stratify_cols: Sequence[str],
    random_seed: int,
) -> pd.DataFrame:
    require_columns(registry, REQUIRED_COLUMNS | set(stratify_cols), label="registry")
    if registry["physical_window_id"].duplicated().any():
        raise ValueError("registry 中 physical_window_id 不唯一")

    if not stratify_cols:
        sampled = registry.sample(n=target_rows, random_state=random_seed, replace=False)
        return sampled.sort_index().reset_index(drop=True)

    group_sizes = registry.groupby(list(stratify_cols), dropna=False, sort=True).size()
    group_counts = _allocate_stratified_counts(group_sizes, target_rows)
    sampled_parts: list[pd.DataFrame] = []
    grouped = registry.groupby(list(stratify_cols), dropna=False, sort=True)
    for group_idx, (key, group) in enumerate(grouped):
        n = int(group_counts.loc[key])
        if n == 0:
            continue
        sampled_parts.append(group.sample(n=n, random_state=random_seed + group_idx, replace=False))
    if not sampled_parts:
        raise ValueError("分层抽样结果为空")
    sampled = pd.concat(sampled_parts, axis=0).sort_index().reset_index(drop=True)
    if len(sampled) != target_rows:
        raise ValueError(f"分层抽样行数错误：expected={target_rows}, actual={len(sampled)}")
    return sampled


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame.groupby(column, dropna=False)["physical_window_id"].nunique().sort_index().items()
    }


def _single_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).unique().tolist()
    return values[0] if len(values) == 1 else None


def build_subset_manifest(
    *,
    input_registry_dir: Path,
    source_registry: pd.DataFrame,
    sampled_registry: pd.DataFrame,
    sample_set_id: str,
    target_rows: int,
    stratify_cols: Sequence[str],
    random_seed: int,
    input_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    input_manifest = dict(input_manifest or {})
    base_sample_set_id = _single_value(source_registry, "sample_set_id") or input_manifest.get("sample_set_id")
    base_registry_id = _single_value(source_registry, "base_registry_id") or input_manifest.get("base_registry_id")
    return {
        "stage": "canonical_expert_matrix_registry_subset",
        "input_registry_dir": str(input_registry_dir),
        "sample_set_id": sample_set_id,
        "base_sample_set_id": base_sample_set_id,
        "base_registry_id": base_registry_id,
        "target_rows": int(target_rows),
        "selected_rows": int(len(sampled_registry)),
        "stratify_cols": list(stratify_cols),
        "random_seed": int(random_seed),
        "split_window_counts": _counts(sampled_registry, "split"),
        "subset_window_counts": _counts(sampled_registry, "subset"),
        "cell_window_counts": _counts(sampled_registry, "official_tsf_cell"),
        "unique_items": int(sampled_registry[["subset", "item_id"]].drop_duplicates().shape[0]),
        "unique_channels": sorted(sampled_registry["channel"].astype(str).unique().tolist()),
        "source_total_rows": int(len(source_registry)),
    }


def materialize_registry_subset(
    *,
    input_registry_dir: Path,
    output_registry_dir: Path,
    sample_set_id: str,
    target_rows: int,
    stratify_cols: Sequence[str],
    random_seed: int,
    overwrite: bool = False,
) -> dict[str, object]:
    input_registry_dir = Path(input_registry_dir)
    output_registry_dir = Path(output_registry_dir)
    if output_registry_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output_registry_dir}")
        shutil.rmtree(output_registry_dir)

    registry_path = input_registry_dir / "window_index.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"缺少输入 registry：{registry_path}")
    source_registry = pd.read_csv(registry_path)
    sampled_registry = _sample_registry(
        source_registry,
        target_rows=target_rows,
        stratify_cols=list(stratify_cols),
        random_seed=random_seed,
    )
    sampled_registry = sampled_registry.copy()
    sampled_registry["sample_set_id"] = sample_set_id

    input_manifest = load_json_manifest(input_registry_dir / "manifest.json")
    manifest = build_subset_manifest(
        input_registry_dir=input_registry_dir,
        source_registry=source_registry,
        sampled_registry=sampled_registry,
        sample_set_id=sample_set_id,
        target_rows=target_rows,
        stratify_cols=list(stratify_cols),
        random_seed=random_seed,
        input_manifest=input_manifest,
    )

    output_registry_dir.mkdir(parents=True, exist_ok=False)
    sampled_registry.to_csv(output_registry_dir / "window_index.csv", index=False)
    write_json_manifest(output_registry_dir / "manifest.json", manifest)
    config_path = input_registry_dir / "config.yml"
    if config_path.exists():
        shutil.copy2(config_path, output_registry_dir / "config.yml")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-registry-dir", type=Path, required=True)
    parser.add_argument("--output-registry-dir", type=Path, required=True)
    parser.add_argument("--sample-set-id", type=str, required=True)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--stratify-cols", type=_parse_stratify_cols, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_registry_subset(
        input_registry_dir=args.input_registry_dir,
        output_registry_dir=args.output_registry_dir,
        sample_set_id=args.sample_set_id,
        target_rows=args.target_rows,
        stratify_cols=args.stratify_cols,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(f"[done] output_registry_dir={args.output_registry_dir}")
    print(f"[done] selected_rows={manifest['selected_rows']}")
    print(f"[done] split_window_counts={manifest['split_window_counts']}")
    print(f"[done] subset_window_counts={manifest['subset_window_counts']}")
    print(f"[done] cell_window_counts={manifest['cell_window_counts']}")


if __name__ == "__main__":
    main()
