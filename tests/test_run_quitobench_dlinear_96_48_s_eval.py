from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_quitobench_dlinear_96_48_s_eval import (
    build_command,
    default_quito_root,
    find_latest_best_checkpoint,
    write_eval_config,
)


def test_find_latest_best_checkpoint_prefers_latest_train_version(tmp_path: Path) -> None:
    base = tmp_path / "outputs" / "dlinear" / "96_48_S" / "FINE_TUNE"
    old_ckpt = base / "ver_0" / "checkpoints" / "best_epoch=1_step=10_MAE=0.5.ckpt"
    new_ckpt = base / "ver_1" / "checkpoints" / "best_epoch=2_step=20_MAE=0.4.ckpt"
    old_ckpt.parent.mkdir(parents=True)
    new_ckpt.parent.mkdir(parents=True)
    old_ckpt.write_text("old", encoding="utf-8")
    new_ckpt.write_text("new", encoding="utf-8")

    assert find_latest_best_checkpoint(tmp_path) == new_ckpt


def test_write_eval_config_sets_single_checkpoint_relative_to_quito_root(tmp_path: Path) -> None:
    src = tmp_path / "base.yaml"
    src.write_text("resume:\n  checkpoint_path:\n  - old.ckpt\n", encoding="utf-8")
    ckpt = tmp_path / "outputs" / "dlinear" / "96_48_S" / "FINE_TUNE" / "ver_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    dst = write_eval_config(
        source_config=src,
        checkpoint_path=ckpt,
        output_config=tmp_path / "patched.yaml",
        quito_root=tmp_path,
    )

    patched = OmegaConf.load(dst)
    assert patched.resume.checkpoint_path == [
        "outputs/dlinear/96_48_S/FINE_TUNE/ver_0/checkpoints/best.ckpt"
    ]


def test_build_command_uses_quito_cli_evaluate() -> None:
    command = build_command(
        config_path=Path("outputs/dlinear/96_48_S/eval_configs/eval_best.yaml"),
        num_processes=1,
        use_gpu=1,
        conda_env="quito",
    )

    assert command == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "quito",
        "quito-cli",
        "evaluate",
        "--config_path",
        "outputs/dlinear/96_48_S/eval_configs/eval_best.yaml",
        "--num_processes",
        "1",
        "--use_gpu",
        "1",
    ]


def test_default_quito_root_points_to_repo_quito_directory() -> None:
    assert default_quito_root() == ROOT / "quito"


def test_dry_run_prints_patched_config_and_command() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_quitobench_dlinear_96_48_s_eval.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "checkpoint=" in result.stdout
    assert "patched_config=" in result.stdout
    assert "quito-cli evaluate" in result.stdout
