"""Build a single-class ("litter") YOLO detection dataset from all detector sources.

Every non-ignored box becomes class 0, whatever its type: the detector only has
to find litter; the crop classifier decides what kind it is.

    uv run build-detector-dataset --config datasets.yaml --out data/detector
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from clean_sweep_data.config import BuildConfig, load_config
from clean_sweep_data.pipeline import (
    SPLITS, SourceStats, place_image, prepare_output_dir, valid_records, write_report,
)
from clean_sweep_data.records import ImageRecord

DETECTOR_CLASS = "litter"


def yolo_label_lines(record: ImageRecord) -> list[str]:
    lines = []
    for box in record.boxes:
        cx = (box.x1 + box.x2) / 2 / record.width
        cy = (box.y1 + box.y2) / 2 / record.height
        w = (box.x2 - box.x1) / record.width
        h = (box.y2 - box.y1) / record.height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def build_detector_dataset(config: BuildConfig, out_dir: Path, copy: bool = False) -> list[SourceStats]:
    out_dir = out_dir.resolve()
    prepare_output_dir(out_dir)

    all_stats = []
    for source in config.detector_sources:
        stats = SourceStats(source.name)
        for record, split, _ in valid_records(source, config, stats):
            image_dst = out_dir / "images" / split / f"{record.key}{record.image_path.suffix.lower()}"
            place_image(record.image_path, image_dst, copy)
            label_dst = out_dir / "labels" / split / f"{record.key}.txt"
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.write_text("\n".join(yolo_label_lines(record)))
        all_stats.append(stats)

    data_yaml = {"path": str(out_dir), **{s: f"images/{s}" for s in SPLITS},
                 "names": {0: DETECTOR_CLASS}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    write_report(all_stats, out_dir, "Detector dataset (all boxes -> 'litter')")
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("datasets.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/detector"))
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_detector_dataset(load_config(args.config), args.out, args.copy)


if __name__ == "__main__":
    main()
