"""Dataset build configuration, loaded from training/datasets.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LITTER = "litter"  # detector-only target: definitely litter, type unknown
IGNORE = "ignore"  # not a pickup target: the box is dropped

DETECTION_FORMATS = frozenset({"coco", "yolo", "voc", "labelme", "negatives"})
CLASSIFICATION_FORMATS = frozenset({"folders"})

_SOURCE_KEYS = {"name", "format", "root", "categories", "default"}


class CategoryMapping:
    """Maps a source dataset's category names onto our targets.

    A target is one of the classifier classes, LITTER, or IGNORE.
    """

    def __init__(self, source: str, categories: dict[str, str], default: str | None,
                 classes: tuple[str, ...]):
        allowed = set(classes) | {LITTER, IGNORE}
        for category, target in [*categories.items(), ("<default>", default)]:
            if target is not None and target not in allowed:
                raise ValueError(
                    f"source {source!r}: category {category!r} maps to {target!r}, "
                    f"expected one of {sorted(allowed)}"
                )
        self._source = source
        self._categories = dict(categories)
        self._default = default

    def resolve(self, category: str) -> str:
        if category in self._categories:
            return self._categories[category]
        if self._default is not None:
            return self._default
        raise KeyError(
            f"source {self._source!r}: category {category!r} is not mapped and no default is set"
        )


@dataclass(frozen=True)
class SourceConfig:
    name: str
    format: str
    root: Path
    settings: dict[str, Any]  # format-specific keys, e.g. annotations, images, class_names
    mapping: CategoryMapping

    def path(self, key: str, default: str = ".") -> Path:
        return self.root / self.settings.get(key, default)


@dataclass(frozen=True)
class BuildConfig:
    classes: tuple[str, ...]
    val_fraction: float
    test_fraction: float
    detector_sources: tuple[SourceConfig, ...]
    classifier_sources: tuple[SourceConfig, ...]


def load_config(path: str | Path) -> BuildConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent.resolve()

    classes = tuple(data["classes"])
    split = data.get("split", {})

    def parse_sources(key: str, formats: frozenset[str]) -> tuple[SourceConfig, ...]:
        sources = []
        for raw in data.get(key) or []:
            if raw["format"] not in formats:
                raise ValueError(
                    f"{key}: source {raw['name']!r} has format {raw['format']!r}, "
                    f"expected one of {sorted(formats)}"
                )
            sources.append(SourceConfig(
                name=raw["name"],
                format=raw["format"],
                root=base_dir / raw["root"],
                settings={k: v for k, v in raw.items() if k not in _SOURCE_KEYS},
                mapping=CategoryMapping(
                    raw["name"], raw.get("categories") or {}, raw.get("default"), classes
                ),
            ))
        names = [s.name for s in sources]
        if len(names) != len(set(names)):
            raise ValueError(f"{key}: duplicate source names in {names}")
        return tuple(sources)

    return BuildConfig(
        classes=classes,
        val_fraction=float(split.get("val", 0.1)),
        test_fraction=float(split.get("test", 0.1)),
        detector_sources=parse_sources("detector_sources", DETECTION_FORMATS),
        classifier_sources=parse_sources("classifier_sources", CLASSIFICATION_FORMATS),
    )
