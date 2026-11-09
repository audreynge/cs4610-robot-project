"""Download TACO (official + unofficial annotations and their images).

Uses the 640px thumbnails by default: plenty for a 640px detector, ~10x smaller
than the originals, and in the same orientation as the annotations. About a
third of the images have no thumbnail; their originals are downloaded, shrunk
to 1280px and saved in the orientation their annotations were drawn in.

    uv run download-taco --out data/raw/taco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from clean_sweep_data.download import fetch, fetch_many
from clean_sweep_data.records import same_aspect

# Originals fetched when a thumbnail is missing are shrunk to this; plenty for 640px training.
SHRUNK_MAX_SIDE = 1280

ANNOTATION_URLS = {
    "annotations.json": "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
    "annotations_unofficial.json":
        "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations_unofficial.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/taco"))
    parser.add_argument("--full-res", action="store_true", help="download original resolution images")
    parser.add_argument("--official-only", action="store_true", help="skip the unreviewed unofficial set")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names = ["annotations.json"] if args.official_only else list(ANNOTATION_URLS)

    jobs, annotated_sizes = [], {}
    for name in names:
        fetch(ANNOTATION_URLS[name], args.out / name)
        for image in json.loads((args.out / name).read_text())["images"]:
            thumbnail = None if args.full_res else image.get("flickr_640_url")
            dest = args.out / image["file_name"]
            jobs.append((thumbnail or image["flickr_url"], dest))
            annotated_sizes[dest] = (image["width"], image["height"])

    print(f"Downloading {len(jobs)} TACO images to {args.out}")
    # Normalize each file as it lands (a no-op for thumbnails) so originals never pile up on disk.
    max_side = None if args.full_res else SHRUNK_MAX_SIDE
    failed = fetch_many(jobs, args.workers,
                        lambda dest: match_annotation_frame(dest, annotated_sizes[dest], max_side))
    if failed:
        print(f"{len(failed)} downloads failed; re-run to retry (finished files are kept).")


def match_annotation_frame(path: Path, annotated_size: tuple[int, int],
                           max_side: int | None = None) -> None:
    """Rewrite the file without EXIF rotation, in the orientation its annotations use.

    Some TACO annotations were drawn on the EXIF-rotated (displayed) image and
    some on the raw pixels; keep whichever frame matches the annotated aspect.
    Optionally downscale so the longest side is max_side. Labels are normalized
    against the annotated size, so only the aspect ratio has to match.
    """
    with Image.open(path) as image:
        rotated = image.getexif().get(0x0112, 1) != 1
        too_big = max_side is not None and max(image.size) > max_side
        if not rotated and not too_big:
            return
        upright = ImageOps.exif_transpose(image)
        raw_matches = same_aspect(image.size, annotated_size)
        result = image.copy() if raw_matches and not same_aspect(upright.size, annotated_size) else upright
        result = result.convert("RGB")
    if too_big:
        result.thumbnail((max_side, max_side))
    result.save(path, quality=90)  # saved without EXIF, so every reader sees the same frame


if __name__ == "__main__":
    main()
