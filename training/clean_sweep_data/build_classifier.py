"""Build the crop-classifier dataset (ImageFolder layout: <split>/<class>/*.jpg).

Crops come from detector-source boxes whose target is a class (LITTER boxes have
no known type and are skipped), plus whole images from classifier sources.
All crops from one image share that image's split, so nothing leaks across splits.

    uv run build-classifier-dataset --config datasets.yaml --out data/classifier
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from clean_sweep_data.config import BuildConfig, load_config
from clean_sweep_data.pipeline import SourceStats, prepare_output_dir, valid_records, write_report
from clean_sweep_data.records import Box, open_upright


def padded_crop_box(box: Box, scale: float, pad: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Scale an annotation box to the actual image and pad it by `pad` of its size on each side."""
    x1, y1, x2, y2 = (v * scale for v in (box.x1, box.y1, box.x2, box.y2))
    pad_x, pad_y = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y)),
            min(size[0], round(x2 + pad_x)), min(size[1], round(y2 + pad_y)))


def build_classifier_dataset(config: BuildConfig, out_dir: Path, pad: float = 0.1,
                             min_size: int = 32) -> list[SourceStats]:
    out_dir = out_dir.resolve()
    prepare_output_dir(out_dir)
    for split in ("train", "val", "test"):
        for name in config.classes:
            (out_dir / split / name).mkdir(parents=True, exist_ok=True)

    all_stats = []
    for source in (*config.detector_sources, *config.classifier_sources):
        stats = SourceStats(source.name)
        too_small = 0
        for record, split, actual in valid_records(source, config, stats):
            boxes = [b for b in record.boxes if b.target in config.classes]
            if not boxes:
                continue
            scale = actual[0] / record.width
            image = open_upright(record.image_path)
            for index, box in enumerate(boxes):
                crop_box = padded_crop_box(box, scale, pad, actual)
                if min(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]) < min_size:
                    too_small += 1
                    continue
                dst = out_dir / split / box.target / f"{record.key}_{index}.jpg"
                image.crop(crop_box).save(dst, quality=95)
        if too_small:
            logging.info("%s: skipped %d crops smaller than %dpx", source.name, too_small, min_size)
        all_stats.append(stats)

    write_report(all_stats, out_dir, "Classifier dataset (boxes per target; only class targets are cropped)")
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("datasets.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/classifier"))
    parser.add_argument("--pad", type=float, default=0.1, help="context padding as a fraction of box size")
    parser.add_argument("--min-size", type=int, default=32, help="drop crops smaller than this (px)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_classifier_dataset(load_config(args.config), args.out, args.pad, args.min_size)


if __name__ == "__main__":
    main()
