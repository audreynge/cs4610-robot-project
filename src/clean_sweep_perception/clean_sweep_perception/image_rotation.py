"""Rotate the head camera image upright for the models, and map boxes back.

Stretch's head camera is mounted sideways, so its raw image is rotated 90°.
The models were trained on upright photos, so detection runs on an upright
copy; geometry needs boxes in raw pixels to match the camera intrinsics.
"""

from __future__ import annotations

import numpy as np

from clean_sweep_perception.geometry import Box

# Name -> number of 90° clockwise turns that make the raw image upright.
ROTATIONS = {"none": 0, "cw90": 1, "180": 2, "ccw90": 3}


def _turns(rotation: str) -> int:
    if rotation not in ROTATIONS:
        raise ValueError(f"image rotation {rotation!r} not in {sorted(ROTATIONS)}")
    return ROTATIONS[rotation]


def rotate_upright(image: np.ndarray, rotation: str) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=-_turns(rotation)))  # rot90 turns counter-clockwise


def box_to_raw(box: Box, rotation: str, raw_shape: tuple[int, ...]) -> Box:
    """Map a box from upright-image pixels back to raw-image pixels.

    Coordinates are continuous (box edges), so a pixel [u, u+1) maps exactly.
    """
    raw_h, raw_w = raw_shape[:2]
    turns = _turns(rotation)
    corners = [(box.x1, box.y1), (box.x2, box.y2)]
    if turns == 0:
        mapped = corners
    elif turns == 1:  # raw (u, v) -> upright (raw_h - v, u)
        mapped = [(y, raw_h - x) for x, y in corners]
    elif turns == 2:  # raw (u, v) -> upright (raw_w - u, raw_h - v)
        mapped = [(raw_w - x, raw_h - y) for x, y in corners]
    else:  # raw (u, v) -> upright (v, raw_w - u)
        mapped = [(raw_w - y, x) for x, y in corners]
    (ax, ay), (bx, by) = mapped
    return Box(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
