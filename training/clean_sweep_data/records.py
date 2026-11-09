"""Source-independent representation of an annotated image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels of the owning record's width/height."""

    x1: float
    y1: float
    x2: float
    y2: float
    target: str  # a classifier class or config.LITTER


@dataclass(frozen=True)
class ImageRecord:
    key: str  # unique, filesystem-safe name: "<source>__<path relative to source root>"
    image_path: Path
    width: int  # the image size the box coordinates refer to
    height: int
    boxes: tuple[Box, ...]


def make_key(source: str, root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root).with_suffix("")
    return f"{source}__{'_'.join(relative.parts)}"


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
             target: str) -> Box | None:
    """Clip to the image; return None if nothing usable is left."""
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return Box(x1, y1, x2, y2, target)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) as displayed, i.e. after applying EXIF orientation.

    Ultralytics applies EXIF orientation when loading, so this is the frame
    labels must be expressed in.
    """
    with Image.open(path) as image:
        orientation = image.getexif().get(0x0112, 1)
        width, height = image.size
    # Orientations 5-8 swap width and height.
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def open_upright(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def same_aspect(actual: tuple[int, int], expected: tuple[int, int], tolerance: float = 0.02) -> bool:
    """True if two sizes have the same aspect ratio (resizing is fine, rotation is not)."""
    return abs(actual[0] / actual[1] - expected[0] / expected[1]) <= tolerance * expected[0] / expected[1]
