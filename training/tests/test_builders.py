import json

import yaml
from PIL import Image

from clean_sweep_data.build_classifier import build_classifier_dataset
from clean_sweep_data.build_detector import build_detector_dataset
from clean_sweep_data.config import BuildConfig, LITTER

from conftest import CLASSES, make_image, make_source


def coco_source(root, images, annotations, categories):
    (root / "ann.json").write_text(json.dumps({
        "categories": [{"id": i, "name": n} for i, n in enumerate(categories)],
        "images": images, "annotations": annotations,
    }))
    return make_source(root, "coco", {"Drink can": "can", "Unlabeled litter": LITTER},
                       annotations="ann.json")


def config_with(detector_sources=(), classifier_sources=()):
    # val/test fractions of 0 put everything in train, so file locations are predictable.
    return BuildConfig(CLASSES, 0.0, 0.0, tuple(detector_sources), tuple(classifier_sources))


def test_detector_writes_single_class_labels_and_reports_problems(tmp_path):
    raw = tmp_path / "raw"
    # Annotations are for a 200x100 image; the file on disk is a half-size thumbnail.
    make_image(raw / "a.jpg", (100, 50))
    make_image(raw / "rotated.jpg", (50, 100))  # orientation disagrees with annotations
    source = coco_source(
        raw,
        images=[{"id": 0, "file_name": "a.jpg", "width": 200, "height": 100},
                {"id": 1, "file_name": "rotated.jpg", "width": 200, "height": 100},
                {"id": 2, "file_name": "missing.jpg", "width": 200, "height": 100}],
        annotations=[{"image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 50]},
                     {"image_id": 0, "category_id": 1, "bbox": [100, 50, 100, 50]}],
        categories=["Drink can", "Unlabeled litter"],
    )
    not_downloaded = make_source(tmp_path / "nope", "negatives", name="nope")

    out = tmp_path / "out"
    [stats, skipped] = build_detector_dataset(config_with([source, not_downloaded]), out)

    labels = (out / "labels" / "train" / "src__a.txt").read_text().splitlines()
    assert labels == ["0 0.250000 0.250000 0.500000 0.500000",
                      "0 0.750000 0.750000 0.500000 0.500000"]
    assert (out / "images" / "train" / "src__a.jpg").resolve() == (raw / "a.jpg").resolve()
    assert stats.images["train"] == 1
    assert stats.boxes == {"can": 1, LITTER: 1}
    assert (stats.size_mismatch, stats.missing_images) == (1, 1)
    assert not skipped.downloaded

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "litter"}
    assert data["train"] == "images/train"


def test_classifier_crops_class_boxes_only(tmp_path):
    raw = tmp_path / "raw"
    make_image(raw / "a.jpg", (200, 100))
    source = coco_source(
        raw,
        images=[{"id": 0, "file_name": "a.jpg", "width": 400, "height": 200}],  # 2x the file
        annotations=[{"image_id": 0, "category_id": 0, "bbox": [100, 50, 100, 100]},
                     {"image_id": 0, "category_id": 1, "bbox": [0, 0, 100, 100]},  # litter: no crop
                     {"image_id": 0, "category_id": 0, "bbox": [0, 0, 20, 20]}],  # too small
        categories=["Drink can", "Unlabeled litter"],
    )
    folders_root = tmp_path / "folders"
    make_image(folders_root / "metal" / "m.jpg", (64, 64))
    folders = make_source(folders_root, "folders", {"metal": "can"}, name="tn")

    out = tmp_path / "out"
    build_classifier_dataset(config_with([source], [folders]), out, pad=0.1, min_size=32)

    crops = sorted(p.name for p in (out / "train" / "can").iterdir())
    assert crops == ["src__a_0.jpg", "tn__metal_m_0.jpg"]
    # Box (50,25)-(100,75) in file pixels, padded by 10% (5px) on each side.
    with Image.open(out / "train" / "can" / "src__a_0.jpg") as crop:
        assert crop.size == (60, 60)
    assert not any((out / "train" / "other").iterdir())
