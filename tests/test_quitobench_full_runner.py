from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.full.run_quitobench_full import (
    FullRunSpec,
    _latest_checkpoint,
    _resolve_resume_checkpoint,
    make_full_output_dir,
    build_stage_commands,
    find_best_tune_config,
    write_finetune_config_with_resume,
    write_finetune_config_with_best_params,
)


def test_build_stage_commands_orders_tune_finetune_evaluate() -> None:
    spec = FullRunSpec(model="patchtst", window="96_48_S")

    commands = build_stage_commands(
        spec=spec,
        conda_env="quito",
        tune_processes=2,
        finetune_processes=4,
        evaluate_processes=3,
        use_gpu=1,
        num_samples=7,
        master_port=29511,
        evaluate_config_path=Path("configs/evaluate/patchtst/96_48_S.yaml"),
    )

    assert [command.stage for command in commands] == ["tune", "finetune", "evaluate"]
    assert commands[0].argv == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "quito",
        "python",
        "quito/scripts/tune.py",
        "--config_path=configs/tune/patchtst/96_48_S.yaml",
        "--tuning_config_path=configs/tune/patchtst/tuning_config.yaml",
        "--num_processes=2",
        "--num_samples=7",
        "--use_gpu=1",
    ]
    assert commands[1].argv[:8] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "quito",
        "torchrun",
        "--nproc_per_node=4",
        "--master_port=29511",
    ]
    assert commands[2].argv == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "quito",
        "python",
        "quito/scripts/evaluate.py",
        "--config_path=configs/evaluate/patchtst/96_48_S.yaml",
        "--num_processes=3",
        "--use_gpu=1",
    ]


def test_build_stage_commands_uses_dlinear_finetune_config_as_tune_base() -> None:
    commands = build_stage_commands(
        spec=FullRunSpec(model="dlinear", window="96_48_S"),
        conda_env="quito",
        tune_processes=1,
        finetune_processes=1,
        evaluate_processes=1,
        use_gpu=1,
        num_samples=10,
        master_port=29501,
        evaluate_config_path=Path("configs/evaluate/dlinear/96_48_S.yaml"),
    )

    assert "--config_path=configs/finetune/dlinear/96_48_S.yaml" in commands[0].argv
    assert "--tuning_config_path=../tools/full/dlinear_tuning_config.yaml" in commands[0].argv


def test_wrapper_dry_run_prints_three_ordered_stages() -> None:
    result = subprocess.run(
        [sys.executable, "tools/full/run_patchtst_96_48_S_full.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    tune_index = result.stdout.index("[tune]")
    finetune_index = result.stdout.index("[finetune]")
    evaluate_index = result.stdout.index("[evaluate]")
    assert tune_index < finetune_index < evaluate_index


def test_find_best_tune_config_selects_lowest_best_metric(tmp_path: Path) -> None:
    tune_root = tmp_path / "outputs" / "patchtst" / "96_48_S" / "TUNE" / "ver_0" / "param_tuning"
    trial_a = tune_root / "trial_a"
    trial_b = tune_root / "trial_b"
    trial_a.mkdir(parents=True)
    trial_b.mkdir(parents=True)
    (trial_a / "result.json").write_text(json.dumps({"best_metric": 0.8}), encoding="utf-8")
    (trial_a / "params.json").write_text(json.dumps({"search_space": {"model": {"d_model": 64}}}), encoding="utf-8")
    (trial_b / "result.json").write_text(json.dumps({"best_metric": 0.3}), encoding="utf-8")
    (trial_b / "params.json").write_text(json.dumps({"search_space": {"model": {"d_model": 128}}}), encoding="utf-8")

    best = find_best_tune_config(tmp_path, FullRunSpec(model="patchtst", window="96_48_S"))

    assert best.metric == 0.3
    assert best.config == {"model": {"d_model": 128}}


def test_write_finetune_config_merges_best_params(tmp_path: Path) -> None:
    quito_root = tmp_path / "quito"
    source = quito_root / "configs" / "finetune" / "patchtst"
    source.mkdir(parents=True)
    (source / "96_48_S.yaml").write_text(
        yaml.safe_dump({"model": {"model_name": "PatchTST", "d_model": 64}, "training": {"learning_rate": 0.001}}),
        encoding="utf-8",
    )

    output = write_finetune_config_with_best_params(
        quito_root=quito_root,
        spec=FullRunSpec(model="patchtst", window="96_48_S"),
        best_params={"model": {"d_model": 128}},
        output_dir=tmp_path,
    )

    assert output == tmp_path / "best_finetune_config.yaml"
    merged = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert merged["model"]["model_name"] == "PatchTST"
    assert merged["model"]["d_model"] == 128
    assert merged["training"]["learning_rate"] == 0.001


def test_write_finetune_config_with_resume_sets_checkpoint(tmp_path: Path) -> None:
    quito_root = tmp_path / "quito"
    source = quito_root / "configs" / "finetune" / "patchtst"
    source.mkdir(parents=True)
    (source / "1024_512_S.yaml").write_text(
        yaml.safe_dump({"model": {"model_name": "PatchTST"}, "resume": {"checkpoint_path": None}}),
        encoding="utf-8",
    )
    checkpoint = quito_root / "outputs" / "patchtst" / "1024_512_S" / "FINE_TUNE" / "ver_0" / "checkpoints" / "ckpt.ckpt"

    output = write_finetune_config_with_resume(
        quito_root=quito_root,
        spec=FullRunSpec(model="patchtst", window="1024_512_S"),
        checkpoint_path=checkpoint,
        output_dir=tmp_path,
    )

    assert output == tmp_path / "resume_finetune_config.yaml"
    merged = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert merged["model"]["model_name"] == "PatchTST"
    assert merged["resume"]["checkpoint_path"] == str(checkpoint)


def test_resolve_resume_checkpoint_rejects_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    try:
        _resolve_resume_checkpoint(checkpoint_dir)
    except FileNotFoundError as exc:
        assert "not a checkpoint file" in str(exc)
    else:
        raise AssertionError("directory checkpoint path should fail")


def test_latest_checkpoint_prefers_best_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "outputs" / "patchtst" / "96_48_S" / "FINE_TUNE" / "ver_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    regular = ckpt_dir / "ckpt_epoch=2_step=20.ckpt"
    best = ckpt_dir / "best_epoch=1_step=10_mae=0.123.ckpt"
    regular.write_text("regular", encoding="utf-8")
    best.write_text("best", encoding="utf-8")

    assert _latest_checkpoint(tmp_path, FullRunSpec(model="patchtst", window="96_48_S")) == best


def test_make_full_output_dir_uses_versioned_full_directory(tmp_path: Path) -> None:
    spec = FullRunSpec(model="crossformer", window="576_288_S")

    first = make_full_output_dir(tmp_path, spec)
    second = make_full_output_dir(tmp_path, spec)

    assert first == tmp_path / "outputs" / "crossformer" / "576_288_S" / "FULL" / "ver_0"
    assert second == tmp_path / "outputs" / "crossformer" / "576_288_S" / "FULL" / "ver_1"
    assert first.exists()
    assert second.exists()


def test_wrapper_dry_run_skip_tune_starts_from_finetune() -> None:
    result = subprocess.run(
        [sys.executable, "tools/full/run_crossformer_576_288_S_full.py", "--dry-run", "--skip-tune"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "[tune]" not in result.stdout
    assert "[finetune]" in result.stdout
    assert "configs/finetune/crossformer/576_288_S.yaml" in result.stdout
    assert "[evaluate]" in result.stdout


def test_full_wrappers_use_distinct_default_master_ports() -> None:
    scripts = sorted((ROOT / "tools" / "full").glob("run_*_*_S_full.py"))
    ports = []
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script.relative_to(ROOT)), "--dry-run"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        match = re.search(r"--master_port=(\d+)", result.stdout)
        assert match, result.stdout
        ports.append(match.group(1))

    assert len(ports) == 9
    assert len(set(ports)) == 9
