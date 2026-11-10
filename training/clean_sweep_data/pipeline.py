"""Shared steps for the dataset builders: validation, splitting, output dirs, reporting."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

from clean_sweep_data.adapters import read_source
from clean_sweep_data.config import BuildConfig, SourceConfig
from clean_sweep_data.records import ImageRecord, image_size, same_aspect

log = logging.getLogger(__name__)

MARKER = ".clean_sweep_dataset"
SPLITS = ("train", "val", "test")


@dataclass
class SourceStats:
    name: str
    downloaded: bool = True
    images: Counter = field(default_factory=Counter)  # split -> count
    boxes: Counter = field(default_factory=Counter)  # target -> count
    empty_images: int = 0  # images with no boxes (negatives)
    missing_images: int = 0  # listed in annotations but not on disk
    size_mismatch: int = 0  # file orientation/aspect disagrees with annotations


def assign_split(key: str, val_fraction: float, test_fraction: float) -> str:
    """Deterministic split from a hash of the key, stable across rebuilds."""
    bucket = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def valid_records(source: SourceConfig, config: BuildConfig,
                  stats: SourceStats) -> Iterator[tuple[ImageRecord, str, tuple[int, int]]]:
    """Yield (record, split, actual_size) for records whose image exists and matches.

    Updates `stats` as a side effect. Sources that aren't downloaded are skipped.
    """
    if not source.root.exists():
        stats.downloaded = False
        log.warning("%s: %s not found, skipping (see training/README.md)", source.name, source.root)
        return

    for record in read_source(source):
        if not record.image_path.is_file():
            stats.missing_images += 1
            continue
        actual = image_size(record.image_path)
        if not same_aspect(actual, (record.width, record.height)):
            stats.size_mismatch += 1
            continue
        split = assign_split(record.key, config.val_fraction, config.test_fraction)
        stats.images[split] += 1
        stats.boxes.update(box.target for box in record.boxes)
        if not record.boxes:
            stats.empty_images += 1
        yield record, split, actual


def prepare_output_dir(out_dir: Path) -> None:
    """Create an empty output dir, only ever wiping one this tool created."""
    if out_dir.exists() and any(out_dir.iterdir()):
        if not (out_dir / MARKER).exists():
            raise SystemExit(f"{out_dir} is not empty and was not created by this tool; refusing to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MARKER).touch()


def place_image(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_report(stats: list[SourceStats], out_dir: Path, title: str) -> None:
    # Not dataclasses.asdict: it rebuilds Counters from (key, value) pairs, corrupting them.
    rows = [{f.name: dict(v) if isinstance(v := getattr(s, f.name), Counter) else v for f in fields(s)}
            for s in stats]
    (out_dir / "report.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{title}")
    header = f"{'source':<16}{'train':>8}{'val':>8}{'test':>8}{'empty':>8}{'missing':>9}{'mismatch':>10}  boxes"
    print(header)
    print("-" * len(header))
    for s in stats:
        if not s.downloaded:
            print(f"{s.name:<16}  (not downloaded, skipped)")
            continue
        boxes = ", ".join(f"{k}={v}" for k, v in sorted(s.boxes.items()))
        print(f"{s.name:<16}{s.images['train']:>8}{s.images['val']:>8}{s.images['test']:>8}"
              f"{s.empty_images:>8}{s.missing_images:>9}{s.size_mismatch:>10}  {boxes}")
    print(f"\nWrote {out_dir}")
