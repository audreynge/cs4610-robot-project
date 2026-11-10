from collections import Counter

import pytest

from clean_sweep_data.pipeline import MARKER, assign_split, prepare_output_dir
from clean_sweep_data.records import image_size, same_aspect

from conftest import make_image


def test_split_is_deterministic_and_roughly_proportional():
    keys = [f"src__img_{i}" for i in range(5000)]
    splits = [assign_split(k, 0.1, 0.1) for k in keys]
    assert splits == [assign_split(k, 0.1, 0.1) for k in keys]
    counts = Counter(splits)
    assert 0.07 < counts["val"] / len(keys) < 0.13
    assert 0.07 < counts["test"] / len(keys) < 0.13


def test_image_size_applies_exif_rotation(tmp_path):
    assert image_size(make_image(tmp_path / "plain.jpg", (100, 50))) == (100, 50)
    assert image_size(make_image(tmp_path / "rotated.jpg", (100, 50), exif_orientation=6)) == (50, 100)


def test_same_aspect_allows_resizing_but_not_rotation():
    assert same_aspect((480, 640), (1537, 2049))
    assert not same_aspect((640, 480), (1537, 2049))


def test_prepare_output_dir_refuses_foreign_directory(tmp_path):
    (tmp_path / "precious.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path)
    assert (tmp_path / "precious.txt").exists()


def test_prepare_output_dir_wipes_its_own_output(tmp_path):
    out = tmp_path / "out"
    prepare_output_dir(out)
    (out / "stale.txt").write_text("old")
    prepare_output_dir(out)
    assert sorted(p.name for p in out.iterdir()) == [MARKER]
