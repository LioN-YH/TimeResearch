from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.workspace_move_manifest_check import check_move_manifest


def test_check_move_manifest_reports_ready_path_state(tmp_path: Path) -> None:
    source = tmp_path / "old"
    source.mkdir()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "status": "ready",
                "category": "smoke",
                "old_path": str(source),
                "new_path": str(tmp_path / "archive" / "old"),
                "reason": "toy",
            }
        ]
    ).to_csv(manifest, index=False)

    result = check_move_manifest(manifest)

    assert result.loc[0, "old_exists"] is True
    assert result.loc[0, "new_exists"] is False
    assert result.loc[0, "ready_to_move"] is True


def test_check_move_manifest_blocks_missing_source_and_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "archive" / "old"
    target.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "status": "ready",
                "category": "smoke",
                "old_path": str(tmp_path / "missing"),
                "new_path": str(target),
                "reason": "toy",
            }
        ]
    ).to_csv(manifest, index=False)

    result = check_move_manifest(manifest)

    assert result.loc[0, "old_exists"] is False
    assert result.loc[0, "new_exists"] is True
    assert result.loc[0, "ready_to_move"] is False


def test_check_move_manifest_ignores_hold_rows_for_move_readiness(tmp_path: Path) -> None:
    source = tmp_path / "keep"
    source.mkdir()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "status": "hold",
                "category": "canonical",
                "old_path": str(source),
                "new_path": "",
                "reason": "keep",
            }
        ]
    ).to_csv(manifest, index=False)

    result = check_move_manifest(manifest)

    assert result.loc[0, "old_exists"] is True
    assert result.loc[0, "new_exists"] is False
    assert result.loc[0, "ready_to_move"] is False


def test_check_move_manifest_validates_moved_rows(tmp_path: Path) -> None:
    target = tmp_path / "archive" / "old"
    target.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "status": "moved",
                "category": "smoke",
                "old_path": str(tmp_path / "old"),
                "new_path": str(target),
                "reason": "moved",
            }
        ]
    ).to_csv(manifest, index=False)

    result = check_move_manifest(manifest)

    assert result.loc[0, "moved_state_valid"] is True
    assert result.loc[0, "blocked_reason"] == ""
