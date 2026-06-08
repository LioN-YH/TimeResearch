from pathlib import Path

from omegaconf import OmegaConf

from tools.quitobench_quito_native_utils import (
    aggregate_metric_sums,
    build_sample_indices,
    clear_resume_checkpoint_paths,
    find_latest_checkpoint,
)


def test_aggregate_metric_sums_uses_sample_weighted_average() -> None:
    metric_sums: dict[str, float] = {}

    aggregate_metric_sums(metric_sums, {"mse": 2.0, "mae": 1.0}, batch_size=4)
    aggregate_metric_sums(metric_sums, {"mse": 10.0, "mae": 5.0}, batch_size=2)

    assert metric_sums == {"mse": 28.0, "mae": 14.0}


def test_find_latest_checkpoint_prefers_newest_checkpoint_file(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "outputs" / "patchtst" / "96_48_S" / "FINE_TUNE" / "ver_1" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    old_ckpt = ckpt_dir / "ckpt_epoch=0_step=10_mae=3.000.ckpt"
    new_ckpt = ckpt_dir / "ckpt_epoch=1_step=20_mae=2.000.ckpt"
    old_ckpt.write_bytes(b"old")
    new_ckpt.write_bytes(b"new")

    assert find_latest_checkpoint(tmp_path / "outputs" / "patchtst" / "96_48_S") == new_ckpt


def test_build_sample_indices_applies_stride_and_max_samples() -> None:
    assert build_sample_indices(dataset_len=20, stride=3, max_samples=4) == [0, 3, 6, 9]


def test_build_sample_indices_uses_full_dataset_when_no_limits() -> None:
    assert build_sample_indices(dataset_len=5, stride=1, max_samples=None) == [0, 1, 2, 3, 4]


def test_clear_resume_checkpoint_paths_removes_evaluate_checkpoint_lists() -> None:
    config = OmegaConf.create(
        {
            "model": {"checkpoint_path": ["./models/a.ckpt"]},
            "resume": {"checkpoint_path": ["./models/b.ckpt"]},
        }
    )

    clear_resume_checkpoint_paths(config)

    assert config.model.checkpoint_path is None
    assert config.resume.checkpoint_path is None
