from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_quitobench_patchtst_96_48_s_cli import build_command, default_quito_root


def test_build_command_uses_official_patchtst_96_48_s_config() -> None:
    command = build_command(
        config_path=Path("configs/finetune/patchtst/96_48_S.yaml"),
        num_processes=1,
        use_gpu=1,
        conda_env="quito",
        master_port=29501,
    )

    assert command == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "quito",
        "torchrun",
        "--nproc_per_node=1",
        "--master_port=29501",
        "quito/scripts/finetune.py",
        "--use_gpu=1",
        "--config_path=configs/finetune/patchtst/96_48_S.yaml",
    ]


def test_default_quito_root_points_to_repo_quito_directory() -> None:
    assert default_quito_root() == ROOT / "quito"


def test_dry_run_prints_command_without_running_training() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_quitobench_patchtst_96_48_s_cli.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "cd " in result.stdout
    assert "torchrun" in result.stdout
    assert "--master_port=29501" in result.stdout
    assert "configs/finetune/patchtst/96_48_S.yaml" in result.stdout
