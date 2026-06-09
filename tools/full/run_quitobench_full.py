"""Run QuitoBench full neural-model protocol: tune, finetune, evaluate.

The protocol order is intentionally fixed. For trained neural models on
QuitoBench, run hyperparameter search first, then finetune with the best trial
hyperparameters, then evaluate the best finetune checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs.
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("patchtst", "dlinear", "crossformer")
WINDOWS = ("96_48_S", "576_288_S", "1024_512_S")


@dataclass(frozen=True)
class FullRunSpec:
    model: str
    window: str


@dataclass(frozen=True)
class StageCommand:
    stage: str
    argv: list[str]


@dataclass(frozen=True)
class BestTuneConfig:
    metric: float
    trial_dir: Path
    config: dict


def default_quito_root() -> Path:
    return ROOT / "quito"


def _relative_config(stage: str, model: str, window: str) -> Path:
    if stage == "tune" and model == "dlinear":
        return Path("configs") / "finetune" / model / f"{window}.yaml"
    return Path("configs") / stage / model / f"{window}.yaml"


def _tuning_config(model: str) -> Path:
    if model == "dlinear":
        return Path("..") / "tools" / "full" / "dlinear_tuning_config.yaml"
    return Path("configs") / "tune" / model / "tuning_config.yaml"


def validate_config_paths(quito_root: Path, spec: FullRunSpec) -> None:
    missing = [
        rel_path
        for rel_path in (
            _relative_config("tune", spec.model, spec.window),
            _tuning_config(spec.model),
            _relative_config("finetune", spec.model, spec.window),
            _relative_config("evaluate", spec.model, spec.window),
        )
        if not (quito_root / rel_path).exists()
    ]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing Quito config path(s):\n{joined}")


def make_full_output_dir(quito_root: Path, spec: FullRunSpec) -> Path:
    base = quito_root / "outputs" / spec.model / spec.window / "FULL"
    index = 0
    while True:
        candidate = base / f"ver_{index}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1


def build_stage_commands(
    *,
    spec: FullRunSpec,
    conda_env: str,
    tune_processes: int,
    finetune_processes: int,
    evaluate_processes: int,
    use_gpu: int,
    num_samples: int,
    master_port: int,
    evaluate_config_path: Path,
    finetune_config_path: Path | None = None,
) -> list[StageCommand]:
    tune_config = _relative_config("tune", spec.model, spec.window)
    finetune_config = finetune_config_path or _relative_config("finetune", spec.model, spec.window)
    tuning_config = _tuning_config(spec.model)

    conda_prefix = ["conda", "run", "--no-capture-output", "-n", conda_env]
    return [
        StageCommand(
            stage="tune",
            argv=[
                *conda_prefix,
                "python",
                "quito/scripts/tune.py",
                f"--config_path={tune_config}",
                f"--tuning_config_path={tuning_config}",
                f"--num_processes={tune_processes}",
                f"--num_samples={num_samples}",
                f"--use_gpu={use_gpu}",
            ],
        ),
        StageCommand(
            stage="finetune",
            argv=[
                *conda_prefix,
                "torchrun",
                f"--nproc_per_node={finetune_processes}",
                f"--master_port={master_port}",
                "quito/scripts/finetune.py",
                f"--use_gpu={use_gpu}",
                f"--config_path={finetune_config}",
            ],
        ),
        StageCommand(
            stage="evaluate",
            argv=[
                *conda_prefix,
                "python",
                "quito/scripts/evaluate.py",
                f"--config_path={evaluate_config_path}",
                f"--num_processes={evaluate_processes}",
                f"--use_gpu={use_gpu}",
            ],
        ),
    ]


def _read_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    lines = [line for line in text.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _extract_search_space(trial_dir: Path, result: dict) -> dict:
    config = result.get("config")
    if isinstance(config, dict) and isinstance(config.get("search_space"), dict):
        return config["search_space"]

    params_path = trial_dir / "params.json"
    if params_path.exists():
        params = _read_json(params_path)
        if isinstance(params.get("search_space"), dict):
            return params["search_space"]
        if isinstance(params.get("config"), dict) and isinstance(params["config"].get("search_space"), dict):
            return params["config"]["search_space"]

    raise FileNotFoundError(f"Cannot find Ray Tune search_space in {trial_dir}")


def _deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def find_best_tune_config(quito_root: Path, spec: FullRunSpec) -> BestTuneConfig:
    tune_root = quito_root / "outputs" / spec.model / spec.window / "TUNE"
    result_paths = sorted(tune_root.glob("ver_*/param_tuning/**/result.json"))
    candidates: list[BestTuneConfig] = []
    for result_path in result_paths:
        result = _read_json(result_path)
        if "best_metric" not in result:
            continue
        metric = float(result["best_metric"])
        trial_dir = result_path.parent
        candidates.append(
            BestTuneConfig(
                metric=metric,
                trial_dir=trial_dir,
                config=_extract_search_space(trial_dir, result),
            )
        )

    if not candidates:
        raise FileNotFoundError(f"No Ray Tune trial result with best_metric found under {tune_root}")
    return min(candidates, key=lambda candidate: candidate.metric)


def write_finetune_config_with_best_params(
    *, quito_root: Path, spec: FullRunSpec, best_params: dict, output_dir: Path
) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write a finetune config with best hyperparameters")

    source = quito_root / _relative_config("finetune", spec.model, spec.window)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    merged = _deep_update(config, best_params)

    target = output_dir / "best_finetune_config.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return target


def _write_finetune_config_with_resume_from_source(*, source: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write a finetune config with a resume checkpoint")

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("resume", {})["checkpoint_path"] = str(checkpoint_path)

    target = output_dir / "resume_finetune_config.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def write_finetune_config_with_resume(
    *, quito_root: Path, spec: FullRunSpec, checkpoint_path: Path, output_dir: Path
) -> Path:
    source = quito_root / _relative_config("finetune", spec.model, spec.window)
    return _write_finetune_config_with_resume_from_source(
        source=source,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )


def _resolve_resume_checkpoint(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Resume checkpoint is not a checkpoint file: {resolved}")
    return resolved


def _latest_checkpoint(quito_root: Path, spec: FullRunSpec) -> Path | None:
    base = quito_root / "outputs" / spec.model / spec.window / "FINE_TUNE"
    if not base.exists():
        return None
    candidates = sorted(base.glob("ver_*/checkpoints/best_*.ckpt"), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]
    candidates = sorted(base.glob("ver_*/checkpoints/ckpt_*.ckpt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _write_eval_config_with_checkpoint(
    *, quito_root: Path, spec: FullRunSpec, checkpoint_path: Path, output_dir: Path
) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write an evaluate config with the finetune checkpoint")

    source = quito_root / _relative_config("evaluate", spec.model, spec.window)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("resume", {})["checkpoint_path"] = [str(checkpoint_path)]

    target = output_dir / "evaluate_best_checkpoint.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def run_commands(commands: Sequence[StageCommand], quito_root: Path, dry_run: bool) -> int:
    print("[run] cd " + str(quito_root), flush=True)
    for command in commands:
        print(f"[{command.stage}] " + shlex.join(command.argv), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(command.argv, cwd=quito_root, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--window", choices=WINDOWS, required=True)
    parser.add_argument("--quito-root", type=Path, default=default_quito_root())
    parser.add_argument("--conda-env", default="quito")
    parser.add_argument("--tune-processes", type=int, default=1)
    parser.add_argument("--finetune-processes", type=int, default=1)
    parser.add_argument("--evaluate-processes", type=int, default=1)
    parser.add_argument("--use-gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument(
        "--evaluate-checkpoint",
        choices=("best-finetune", "latest-finetune", "config"),
        default="best-finetune",
        help="Use the best finetune checkpoint for evaluate, or keep the checkpoint paths in the evaluate YAML. latest-finetune is kept as a compatibility alias for best-finetune.",
    )
    parser.add_argument("--skip-tune", action="store_true", help="Skip Ray Tune and use configs/finetune/{model}/{window}.yaml directly.")
    parser.add_argument(
        "--resume-finetune-checkpoint",
        type=Path,
        help="Resume the finetune stage from this Quito checkpoint by writing a temporary finetune YAML.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = FullRunSpec(model=args.model, window=args.window)
    quito_root = args.quito_root.resolve()
    validate_config_paths(quito_root, spec)

    if args.dry_run or args.evaluate_checkpoint == "config":
        evaluate_config_path = _relative_config("evaluate", spec.model, spec.window)
        commands = build_stage_commands(
            spec=spec,
            conda_env=args.conda_env,
            tune_processes=args.tune_processes,
            finetune_processes=args.finetune_processes,
            evaluate_processes=args.evaluate_processes,
            use_gpu=args.use_gpu,
            num_samples=args.num_samples,
            master_port=args.master_port,
            evaluate_config_path=evaluate_config_path,
        )
        if args.skip_tune:
            commands = commands[1:]
        return run_commands(commands, quito_root, args.dry_run)

    full_output_dir = make_full_output_dir(quito_root, spec)
    print(f"[full] output_dir={full_output_dir}", flush=True)

    finetune_config = None
    if args.skip_tune:
        print(f"[tune] skipped; using {_relative_config('finetune', spec.model, spec.window)}", flush=True)
        if args.resume_finetune_checkpoint:
            checkpoint_path = _resolve_resume_checkpoint(args.resume_finetune_checkpoint)
            finetune_config = write_finetune_config_with_resume(
                quito_root=quito_root,
                spec=spec,
                checkpoint_path=checkpoint_path,
                output_dir=full_output_dir,
            )
            print(f"[finetune] resume_checkpoint={checkpoint_path}", flush=True)
            print(f"[finetune] resume_config={finetune_config}", flush=True)
    else:
        tune = build_stage_commands(
            spec=spec,
            conda_env=args.conda_env,
            tune_processes=args.tune_processes,
            finetune_processes=args.finetune_processes,
            evaluate_processes=args.evaluate_processes,
            use_gpu=args.use_gpu,
            num_samples=args.num_samples,
            master_port=args.master_port,
            evaluate_config_path=_relative_config("evaluate", spec.model, spec.window),
        )[:1]
        exit_code = run_commands(tune, quito_root, dry_run=False)
        if exit_code != 0:
            return exit_code

        best_tune = find_best_tune_config(quito_root, spec)
        finetune_config = write_finetune_config_with_best_params(
            quito_root=quito_root,
            spec=spec,
            best_params=best_tune.config,
            output_dir=full_output_dir,
        )
        print(f"[tune] best_metric={best_tune.metric} trial_dir={best_tune.trial_dir}", flush=True)
        print(f"[tune] best_finetune_config={finetune_config}", flush=True)
        if args.resume_finetune_checkpoint:
            checkpoint_path = _resolve_resume_checkpoint(args.resume_finetune_checkpoint)
            finetune_config = _write_finetune_config_with_resume_from_source(
                source=finetune_config,
                checkpoint_path=checkpoint_path,
                output_dir=full_output_dir,
            )
            print(f"[finetune] resume_checkpoint={checkpoint_path}", flush=True)
            print(f"[finetune] resume_config={finetune_config}", flush=True)

    finetune = build_stage_commands(
        spec=spec,
        conda_env=args.conda_env,
        tune_processes=args.tune_processes,
        finetune_processes=args.finetune_processes,
        evaluate_processes=args.evaluate_processes,
        use_gpu=args.use_gpu,
        num_samples=args.num_samples,
        master_port=args.master_port,
        evaluate_config_path=_relative_config("evaluate", spec.model, spec.window),
        finetune_config_path=finetune_config,
    )[1:2]
    exit_code = run_commands(finetune, quito_root, dry_run=False)
    if exit_code != 0:
        return exit_code

    checkpoint = _latest_checkpoint(quito_root, spec)
    if checkpoint is None:
        raise FileNotFoundError(f"No finetune checkpoint found under {quito_root / 'outputs' / spec.model / spec.window}")

    eval_config = _write_eval_config_with_checkpoint(
        quito_root=quito_root,
        spec=spec,
        checkpoint_path=checkpoint,
        output_dir=full_output_dir,
    )
    print(f"[finetune] selected_checkpoint={checkpoint}", flush=True)
    print(f"[evaluate] best_checkpoint_config={eval_config}", flush=True)
    evaluate = build_stage_commands(
        spec=spec,
        conda_env=args.conda_env,
        tune_processes=args.tune_processes,
        finetune_processes=args.finetune_processes,
        evaluate_processes=args.evaluate_processes,
        use_gpu=args.use_gpu,
        num_samples=args.num_samples,
        master_port=args.master_port,
        evaluate_config_path=eval_config,
    )[2:]
    return run_commands(evaluate, quito_root, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
