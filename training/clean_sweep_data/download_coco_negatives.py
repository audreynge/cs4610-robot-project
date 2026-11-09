"""Download indoor COCO val2017 images that contain no litter-like objects.

They are added to the detector dataset with no boxes, teaching it what an
indoor scene *without* litter looks like (furniture, floors, shoes, bags...).

    uv run download-coco-negatives --out data/raw/coco_negatives
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from clean_sweep_data.download import fetch, fetch_many

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
INSTANCES_MEMBER = "annotations/instances_val2017.json"

# Images must show at least one of these, so they are indoor scenes.
INDOOR_CATEGORIES = {
    "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "microwave",
    "oven", "refrigerator", "sink", "potted plant",
}
# ...and none of these, which look like (or are often next to) litter.
LITTER_LIKE_CATEGORIES = {
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
}


def select_negative_images(coco: dict, max_images: int, seed: int = 0) -> list[dict]:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    categories_by_image: dict[int, set[str]] = {}
    for annotation in coco["annotations"]:
        categories_by_image.setdefault(annotation["image_id"], set()).add(names[annotation["category_id"]])

    selected = [
        image for image in coco["images"]
        if (cats := categories_by_image.get(image["id"], set())) & INDOOR_CATEGORIES
        and not cats & LITTER_LIKE_CATEGORIES
    ]
    random.Random(seed).shuffle(selected)
    return sorted(selected[:max_images], key=lambda image: image["file_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/coco_negatives"))
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    selection_file = args.out / "selected_images.json"
    if not selection_file.exists():
        archive = args.out / "annotations_trainval2017.zip"
        print(f"Downloading COCO annotations (~250 MB) to {archive}")
        fetch(ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            coco = json.loads(zf.read(INSTANCES_MEMBER))
        selected = select_negative_images(coco, args.max_images)
        selection_file.write_text(json.dumps(selected, indent=1))
        archive.unlink()  # only the selection is needed from here on

    selected = json.loads(selection_file.read_text())
    print(f"Downloading {len(selected)} negative images to {args.out / 'images'}")
    failed = fetch_many([(image["coco_url"], args.out / "images" / image["file_name"]) for image in selected],
                        args.workers)
    if failed:
        print(f"{len(failed)} downloads failed; re-run to retry.")


if __name__ == "__main__":
    main()
