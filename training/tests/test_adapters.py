import json

import pytest

from clean_sweep_data.adapters import read_source
from clean_sweep_data.config import LITTER
from clean_sweep_data.records import Box

from conftest import make_image, make_source


def test_coco_maps_categories_drops_ignored_and_clips(tmp_path):
    make_image(tmp_path / "batch_1" / "a.jpg", (100, 50))
    (tmp_path / "ann.json").write_text(json.dumps({
        "categories": [{"id": 0, "name": "Drink can"}, {"id": 1, "name": "Cigarette"},
                       {"id": 2, "name": "Unlabeled litter"}],
        "images": [{"id": 7, "file_name": "batch_1/a.jpg", "width": 100, "height": 50}],
        "annotations": [
            {"image_id": 7, "category_id": 0, "bbox": [10, 10, 20, 20]},
            {"image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"image_id": 7, "category_id": 2, "bbox": [90, 40, 30, 30]},  # runs off the image
        ],
    }))
    source = make_source(tmp_path, "coco", {"Drink can": "can", "Cigarette": "ignore",
                                            "Unlabeled litter": LITTER}, annotations="ann.json")

    [record] = list(read_source(source))
    assert record.key == "src__batch_1_a"
    assert record.boxes == (Box(10, 10, 30, 30, "can"), Box(90, 40, 100, 50, LITTER))


def test_coco_reads_multiple_annotation_files(tmp_path):
    for name, file_name in (("a.json", "x/1.jpg"), ("b.json", "y/1.jpg")):
        (tmp_path / name).write_text(json.dumps({
            "categories": [], "annotations": [],
            "images": [{"id": 0, "file_name": file_name, "width": 10, "height": 10}],
        }))
    source = make_source(tmp_path, "coco", annotations=["a.json", "b.json"])
    assert [r.key for r in read_source(source)] == ["src__x_1", "src__y_1"]


def test_yolo_converts_normalized_boxes_and_skips_unlabeled(tmp_path):
    make_image(tmp_path / "imgs" / "a.jpg", (200, 100))
    make_image(tmp_path / "imgs" / "unlabeled.jpg", (200, 100))
    (tmp_path / "imgs" / "a.txt").write_text("1 0.5 0.5 0.1 0.2\n\n")
    source = make_source(tmp_path, "yolo", {"Glass": "bottle", "AluCan": "can"},
                         images="imgs", class_names=["Glass", "AluCan"])

    [record] = list(read_source(source))
    assert (record.width, record.height) == (200, 100)
    assert record.boxes == (Box(90, 40, 110, 60, "can"),)


def test_voc_reads_objects(tmp_path):
    make_image(tmp_path / "JPEGImages" / "a.jpg", (64, 48))
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Annotations" / "a.xml").write_text("""
<annotation><filename>a.jpg</filename><size><width>64</width><height>48</height></size>
  <object><name>paper</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
  <object><name>cap</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>""")
    source = make_source(tmp_path, "voc", {"paper": "paper", "cap": "ignore"},
                         annotations="Annotations", images="JPEGImages")

    [record] = list(read_source(source))
    assert record.boxes == (Box(1, 2, 30, 40, "paper"),)


def test_voc_unmapped_category_raises(tmp_path):
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Annotations" / "a.xml").write_text(
        "<annotation><filename>a.jpg</filename><size><width>9</width><height>9</height></size>"
        "<object><name>mystery</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax>"
        "</bndbox></object></annotation>")
    source = make_source(tmp_path, "voc", annotations="Annotations", images="JPEGImages")
    with pytest.raises(KeyError, match="mystery"):
        list(read_source(source))


def test_negatives_have_no_boxes(tmp_path):
    make_image(tmp_path / "images" / "n.jpg")
    [record] = list(read_source(make_source(tmp_path, "negatives", images="images")))
    assert record.boxes == ()


def test_folders_give_full_image_boxes_and_skip_ignored(tmp_path):
    make_image(tmp_path / "metal" / "m.jpg", (40, 30))
    make_image(tmp_path / "glass" / "g.jpg")
    source = make_source(tmp_path, "folders", {"metal": "can", "glass": "ignore"})

    [record] = list(read_source(source))
    assert record.boxes == (Box(0, 0, 40, 30, "can"),)
