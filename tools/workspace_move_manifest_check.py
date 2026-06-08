"""Check workspace cleanup move manifest without moving files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST = Path("docs/WORKSPACE_MOVE_MANIFEST.csv")
REQUIRED_COLUMNS = {"status", "category", "old_path", "new_path", "reason"}


def _path_exists(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).exists()


def check_move_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """Return per-row readiness checks for a cleanup move manifest."""

    manifest = pd.read_csv(manifest_path).fillna("")
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest 缺少列：{sorted(missing)}")

    result = manifest.copy()
    result["old_exists"] = result["old_path"].map(_path_exists).astype(object)
    result["new_exists"] = result["new_path"].map(_path_exists).astype(object)
    result["ready_to_move"] = (
        result["status"].astype(str).eq("ready")
        & result["old_exists"].astype(bool)
        & ~result["new_exists"].astype(bool)
        & result["new_path"].astype(str).str.len().gt(0)
    ).astype(object)
    result["moved_state_valid"] = (
        result["status"].astype(str).eq("moved")
        & ~result["old_exists"].astype(bool)
        & result["new_exists"].astype(bool)
        & result["new_path"].astype(str).str.len().gt(0)
    ).astype(object)
    result["blocked_reason"] = ""
    ready_rows = result["status"].astype(str).eq("ready")
    result.loc[ready_rows & ~result["old_exists"].astype(bool), "blocked_reason"] = "old_path_missing"
    result.loc[ready_rows & result["new_exists"].astype(bool), "blocked_reason"] = "new_path_exists"
    result.loc[ready_rows & result["new_path"].astype(str).str.len().eq(0), "blocked_reason"] = "new_path_empty"
    moved_rows = result["status"].astype(str).eq("moved")
    result.loc[moved_rows & result["old_exists"].astype(bool), "blocked_reason"] = "moved_old_path_still_exists"
    result.loc[moved_rows & ~result["new_exists"].astype(bool), "blocked_reason"] = "moved_new_path_missing"
    result.loc[moved_rows & result["new_path"].astype(str).str.len().eq(0), "blocked_reason"] = "moved_new_path_empty"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_move_manifest(args.manifest)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        print(f"[done] output_csv={args.output_csv}")
    ready = int(result["ready_to_move"].sum())
    blocked = int((result["status"].astype(str).eq("ready") & ~result["ready_to_move"].astype(bool)).sum())
    invalid_moved = int((result["status"].astype(str).eq("moved") & ~result["moved_state_valid"].astype(bool)).sum())
    hold = int(result["status"].astype(str).eq("hold").sum())
    print(f"[check] ready_to_move={ready}")
    print(f"[check] blocked_ready_rows={blocked}")
    print(f"[check] invalid_moved_rows={invalid_moved}")
    print(f"[check] hold_rows={hold}")
    if blocked or invalid_moved:
        blocked_rows = result[
            (result["status"].astype(str).eq("ready") & ~result["ready_to_move"].astype(bool))
            | (result["status"].astype(str).eq("moved") & ~result["moved_state_valid"].astype(bool))
        ]
        print(blocked_rows[["old_path", "new_path", "blocked_reason"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
