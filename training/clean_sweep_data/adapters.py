"""Readers that turn each public dataset format into ImageRecords.

Every reader applies the source's CategoryMapping: IGNORE boxes are dropped,
everything else keeps its target (a class name or LITTER).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path

from clean_sweep_data.config import IGNORE, LITTER, SourceConfig
from clean_sweep_data.records import Box, ImageRecord, clip_box, image_size, make_key

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_source(source: SourceConfig) -> Iterator[ImageRecord]:
    return _READERS[source.format](source)


def _image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _keep(source: SourceConfig, category: str, coords: tuple[float, float, float, float],
          width: int, height: int) -> Box | None:
    target = source.mapping.resolve(category)
    if target == IGNORE:
        return None
    return clip_box(*coords, width, height, target)


def read_coco(source: SourceConfig) -> Iterator[ImageRecord]:
    """COCO json (e.g. TACO). Settings: annotations (file or list), images (dir)."""
    annotation_files = source.settings.get("annotations", "annotations.json")
    if isinstance(annotation_files, str):
        annotation_files = [annotation_files]
    images_dir = source.path("images")

    for annotation_file in annotation_files:
        coco = json.loads((source.root / annotation_file).read_text())
        category_names = {c["id"]: c["name"] for c in coco["categories"]}
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image in coco["images"]:
            width, height = image["width"], image["height"]
            boxes = []
            for annotation in annotations_by_image.get(image["id"], []):
                x, y, w, h = annotation["bbox"]
                box = _keep(source, category_names[annotation["category_id"]],
                            (x, y, x + w, y + h), width, height)
                if box:
                    boxes.append(box)
            image_path = images_dir / image["file_name"]
            yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                              width, height, tuple(boxes))


def read_yolo(source: SourceConfig) -> Iterator[ImageRecord]:
    """YOLO txt labels. Settings: images (dir), labels (dir, default=images), class_names."""
    images_dir = source.path("images")
    labels_dir = source.path("labels", source.settings.get("images", "."))
    class_names = source.settings["class_names"]

    missing_labels = 0
    for image_path in _image_files(images_dir):
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.exists():
            # An unlabeled image may still contain litter; don't use it as a negative.
            missing_labels += 1
            continue
        width, height = image_size(image_path)
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, w, h = line.split()[:5]
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            box = _keep(source, class_names[int(class_id)],
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), width, height)
            if box:
                boxes.append(box)
        yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                          width, height, tuple(boxes))
    if missing_labels:
        log.warning("%s: skipped %d images without a label file", source.name, missing_labels)


def read_voc(source: SourceConfig) -> Iterator[ImageRecord]:
    """Pascal VOC xml. Settings: annotations (dir of .xml), images (dir)."""
    annotations_dir = source.path("annotations")
    images_dir = source.path("images")

    for xml_path in sorted(annotations_dir.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_path = images_dir / root.findtext("filename", "")
        size = root.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0
        if (not width or not height) and image_path.is_file():
            width, height = image_size(image_path)
        if not width or not height:
            log.warning("%s: no image size for %s, skipping", source.name, xml_path.name)
            continue

        boxes = []
        for obj in root.iter("object"):
            bndbox = obj.find("bndbox")
            coords = tuple(float(bndbox.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            box = _keep(source, obj.findtext("name", "").strip(), coords, width, height)
            if box:
                boxes.append(box)
        yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                          width, height, tuple(boxes))


def read_labelme(source: SourceConfig) -> Iterator[ImageRecord]:
    """LabelMe json, one file per image. Settings: annotations (dir), images (dir, default=annotations).

    Each shape's box is the bounding box of its points, so rectangles and polygons both work.
    """
    annotations_dir = source.path("annotations")
    images_dir = source.path("images", source.settings.get("annotations", "."))

    for json_path in sorted(annotations_dir.rglob("*.json")):
        data = json.loads(json_path.read_text())
        # imagePath is relative to wherever the annotator saved it (often a Windows path), so
        # only its file name is reliable.
        image_path = images_dir / Path(data["imagePath"].replace("\\", "/")).name
        width, height = data["imageWidth"], data["imageHeight"]
        boxes = []
        for shape in data["shapes"]:
            xs, ys = zip(*shape["points"])
            box = _keep(source, shape["label"], (min(xs), min(ys), max(xs), max(ys)), width, height)
            if box:
                boxes.append(box)
        yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                          width, height, tuple(boxes))


def read_negatives(source: SourceConfig) -> Iterator[ImageRecord]:
    """Images known to contain no litter. Settings: images (dir)."""
    images_dir = source.path("images")
    for image_path in _image_files(images_dir):
        width, height = image_size(image_path)
        yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                          width, height, ())


def read_folders(source: SourceConfig) -> Iterator[ImageRecord]:
    """Classification folders (<images>/<category>/*.jpg), one full-image box each.

    Only valid as a classifier source: the box covers the whole image.
    """
    images_dir = source.path("images")
    for category_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        target = source.mapping.resolve(category_dir.name)
        if target in (IGNORE, LITTER):
            continue
        for image_path in _image_files(category_dir):
            width, height = image_size(image_path)
            yield ImageRecord(make_key(source.name, images_dir, image_path), image_path,
                              width, height, (Box(0, 0, width, height, target),))


_READERS: dict[str, Callable[[SourceConfig], Iterator[ImageRecord]]] = {
    "coco": read_coco,
    "yolo": read_yolo,
    "voc": read_voc,
    "labelme": read_labelme,
    "negatives": read_negatives,
    "folders": read_folders,
}
