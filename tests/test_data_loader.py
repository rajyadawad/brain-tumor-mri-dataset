"""Tests for data_loader helpers and config invariants. Uses a tiny on-the-fly
fixture directory so no real dataset is required (CI-safe)."""

import numpy as np
from PIL import Image

from src import config
from src.data_loader import count_images_per_class


def _make_fixture(tmp_path, counts):
    """Create <tmp>/class/img.png files and return the root dir."""
    root = tmp_path / "split"
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = (np.random.rand(8, 8, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{cls}_{i}.png")
    return str(root)


def test_count_images_per_class(tmp_path):
    counts = {"glioma": 3, "notumor": 2}
    root = _make_fixture(tmp_path, counts)
    assert count_images_per_class(root) == counts


def test_count_ignores_non_images(tmp_path):
    root = _make_fixture(tmp_path, {"glioma": 2})
    (tmp_path / "split" / "glioma" / "notes.txt").write_text("ignore me")
    assert count_images_per_class(root) == {"glioma": 2}


def test_count_missing_dir_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        count_images_per_class(str(tmp_path / "does_not_exist"))


def test_config_invariants():
    assert config.IMAGE_SIZE == 224
    assert config.NUM_CLASSES == len(config.CLASS_NAMES) == 4
    assert 0.0 < config.VAL_SPLIT < 1.0
    # DISPLAY_NAMES covers every class.
    assert set(config.DISPLAY_NAMES) == set(config.CLASS_NAMES)
