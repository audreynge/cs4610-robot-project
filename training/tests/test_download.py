import zipfile

import pytest
from PIL import Image

from clean_sweep_data.download import extract_zip
from clean_sweep_data.download_coco_negatives import select_negative_images
from clean_sweep_data.download_taco import match_annotation_frame
from clean_sweep_data.records import image_size

from conftest import make_image


def coco(images_with_categories):
    names = sorted({c for cats in images_with_categories.values() for c in cats})
    ids = {n: i for i, n in enumerate(names)}
    return {
        "categories": [{"id": i, "name": n} for n, i in ids.items()],
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in images_with_categories],
        "annotations": [{"image_id": i, "category_id": ids[c]}
                        for i, cats in images_with_categories.items() for c in cats],
    }


def test_selects_indoor_images_without_litter_like_objects():
    data = coco({
        1: ["chair", "person"],  # indoor, clean -> keep
        2: ["couch", "cup"],  # has a cup -> drop
        3: ["car"],  # outdoor -> drop
        4: [],  # no annotations -> drop
        5: ["bed"],
    })
    selected = select_negative_images(data, max_images=10)
    assert [image["id"] for image in selected] == [1, 5]
    assert len(select_negative_images(data, max_images=1)) == 1


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_keeps_rotated_frame_when_annotations_use_displayed_image(tmp_path):
    path = make_image(tmp_path / "a.jpg", (4000, 3000), exif_orientation=6)  # displays 3000x4000
    match_annotation_frame(path, annotated_size=(3000, 4000), max_side=1280)
    with Image.open(path) as image:
        assert image.size == (960, 1280)
        assert image.getexif().get(0x0112, 1) == 1


def test_keeps_raw_frame_when_annotations_ignore_exif(tmp_path):
    path = make_image(tmp_path / "a.jpg", (4000, 3000), exif_orientation=6)
    match_annotation_frame(path, annotated_size=(4000, 3000), max_side=1280)
    assert image_size(path) == (1280, 960)


def test_full_res_only_strips_rotation(tmp_path):
    path = make_image(tmp_path / "a.jpg", (400, 300), exif_orientation=6)
    match_annotation_frame(path, annotated_size=(300, 400), max_side=None)
    assert image_size(path) == (300, 400)
    with Image.open(path) as image:
        assert image.size == (300, 400)


def test_leaves_small_unrotated_files_untouched(tmp_path):
    small = make_image(tmp_path / "small.jpg", (640, 480))
    before = small.stat().st_mtime_ns
    match_annotation_frame(small, (1280, 960), 1280)
    assert small.stat().st_mtime_ns == before
