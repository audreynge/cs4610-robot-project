"""Ground-truth litter scene for simulation, loaded from YAML (no ROS dependency).

Example file:

    frame_id: map
    objects:
      - label: can
        position: [1.0, 0.5, 0.0]
        width: 0.066
        height: 0.12
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from clean_sweep_perception.labels import LITTER_LABELS


@dataclass(frozen=True)
class StubObject:
    label: str
    position: tuple[float, float, float]
    width: float
    height: float


@dataclass(frozen=True)
class StubScene:
    frame_id: str
    objects: tuple[StubObject, ...]


def load_stub_scene(path: str | Path) -> StubScene:
    """Load and validate a scene file. Raises ValueError on bad content."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    frame_id = data.get("frame_id", "map")
    raw_objects = data.get("objects") or []
    objects = tuple(_parse_object(i, raw) for i, raw in enumerate(raw_objects))
    return StubScene(frame_id=frame_id, objects=objects)


def _parse_object(index: int, raw: dict) -> StubObject:
    where = f"objects[{index}]"

    label = raw.get("label")
    if label not in LITTER_LABELS:
        raise ValueError(f"{where}: label {label!r} not in {LITTER_LABELS}")

    position = raw.get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"{where}: position must be [x, y, z], got {position!r}")

    width = float(raw.get("width", 0))
    height = float(raw.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"{where}: width and height must be positive")

    return StubObject(
        label=label,
        position=tuple(float(v) for v in position),
        width=width,
        height=height,
    )
