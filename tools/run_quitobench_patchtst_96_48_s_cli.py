"""Run official Quito CLI PatchTST 96_48_S finetuning.

This wrapper keeps the official trainer and tqdm progress display intact while
making the working directory, conda environment, and config explicit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path("configs/finetune/patchtst/96_48_S.yaml")


def default_quito_root() -> Path:
    return ROOT / "quito"


def build_command(
    *,
    config_path: Path,
    num_processes: int,
    use_gpu: int,
    conda_env: str,
    master_port: int,
) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "torchrun",
        f"--nproc_per_node={num_processes}",
        f"--master_port={master_port}",
        "quito/scripts/finetune.py",
        f"--use_gpu={use_gpu}",
        f"--config_path={config_path}",
    ]


def _latest_output_dir(quito_root: Path) -> Path | None:
    base = quito_root / "outputs" / "patchtst" / "96_48_S" / "FINE_TUNE"
    if not base.exists():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_files(output_dir: Path | None) -> list[Path]:
    if output_dir is None:
        return []
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quito-root", type=Path, default=default_quito_root())
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--use-gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--conda-env", default="quito")
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quito_root = args.quito_root.resolve()
    command = build_command(
        config_path=args.config_path,
        num_processes=args.num_processes,
        use_gpu=args.use_gpu,
        conda_env=args.conda_env,
        master_port=args.master_port,
    )

    print("[run] cd " + str(quito_root), flush=True)
    print("[run] " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=quito_root, check=False)
    if completed.returncode != 0:
        return completed.returncode

    output_dir = _latest_output_dir(quito_root)
    print("[done] latest_output_dir=" + (str(output_dir) if output_dir else "not_found"), flush=True)
    for checkpoint in _checkpoint_files(output_dir):
        print("[done] checkpoint=" + str(checkpoint), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
