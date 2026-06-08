from pathlib import Path

from omegaconf import OmegaConf

from tools.quitobench_dense_smoke import _patch_config


def _write_two_subset_config(path: Path) -> None:
    config = OmegaConf.create(
        {
            "data": {
                "datasets": {
                    "TEST_DATA_MIN": {"file_name": "open_min_data.parquet"},
                    "TEST_DATA_HOUR": {"file_name": "open_hour_data.parquet"},
                }
            }
        }
    )
    OmegaConf.save(config, path)


def test_patch_config_removes_min_dataset_when_no_min_ids(tmp_path: Path) -> None:
    src = tmp_path / "src.yaml"
    dst = tmp_path / "dst.yaml"
    _write_two_subset_config(src)

    _patch_config(src, dst, hour_ids=[104560], min_ids=[])

    patched = OmegaConf.load(dst)
    assert "TEST_DATA_HOUR" in patched.data.datasets
    assert patched.data.datasets.TEST_DATA_HOUR.ids == [104560]
    assert "TEST_DATA_MIN" not in patched.data.datasets


def test_patch_config_removes_hour_dataset_when_no_hour_ids(tmp_path: Path) -> None:
    src = tmp_path / "src.yaml"
    dst = tmp_path / "dst.yaml"
    _write_two_subset_config(src)

    _patch_config(src, dst, hour_ids=[], min_ids=[11217])

    patched = OmegaConf.load(dst)
    assert "TEST_DATA_MIN" in patched.data.datasets
    assert patched.data.datasets.TEST_DATA_MIN.ids == [11217]
    assert "TEST_DATA_HOUR" not in patched.data.datasets
