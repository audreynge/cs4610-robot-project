from pathlib import Path

from PIL import Image

from clean_sweep_data.config import CategoryMapping, SourceConfig

CLASSES = ("bottle", "can", "cup", "paper", "other")


def make_image(path: Path, size=(100, 50), exif_orientation: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "gray")
    if exif_orientation is None:
        image.save(path)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(path, exif=exif)
    return path


def make_source(root: Path, fmt: str, categories=None, default=None, name="src",
                **settings) -> SourceConfig:
    return SourceConfig(
        name=name, format=fmt, root=root, settings=settings,
        mapping=CategoryMapping(name, categories or {}, default, CLASSES),
    )
