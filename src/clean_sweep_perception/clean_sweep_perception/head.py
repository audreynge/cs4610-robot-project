"""Pan/tilt angles for pointing Stretch's head camera at a point (no ROS)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadLimits:
    """Stretch 2 head joint ranges (rad), from the robot's URDF/calibration."""

    pan_min: float = -4.04
    pan_max: float = 1.73
    tilt_min: float = -1.53  # looking down
    tilt_max: float = 0.79


def look_at_angles(target: np.ndarray, head_origin: np.ndarray, pan_zero_heading: float = 0.0,
                   limits: HeadLimits = HeadLimits()) -> tuple[float, float]:
    """(pan, tilt) that point the camera at `target`.

    Both points are in the base frame (z up). `head_origin` is where the pan and
    tilt axes meet (approximately the `link_head_pan` origin). `pan_zero_heading`
    is the base-frame heading the camera faces at pan = 0. Tilt is negative when
    looking down. Angles outside the joint limits are clamped.
    """
    rel = np.asarray(target, dtype=float) - np.asarray(head_origin, dtype=float)
    pan = math.atan2(rel[1], rel[0]) - pan_zero_heading
    pan = _reachable_pan(pan, limits)
    tilt = math.atan2(rel[2], math.hypot(rel[0], rel[1]))
    return pan, min(max(tilt, limits.tilt_min), limits.tilt_max)


def _reachable_pan(pan: float, limits: HeadLimits) -> float:
    """The equivalent angle (±2π) inside the pan limits, or the nearest limit."""
    candidates = [pan + k * 2 * math.pi for k in (-2, -1, 0, 1, 2)]
    inside = [p for p in candidates if limits.pan_min <= p <= limits.pan_max]
    if inside:
        return min(inside, key=abs)
    return min((limits.pan_min, limits.pan_max),
               key=lambda edge: min(abs(edge - p) for p in candidates))
