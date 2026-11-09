"""3D position and size of a detected object from a box and aligned depth (no ROS).

Frames: the camera optical frame (z forward, x right, y down) and the robot base
frame (z up, floor at z = 0). The caller supplies the camera->base transform
(from tf2) and the color camera intrinsics (from CameraInfo).

Two strategies, tried in order:
1. Depth: deproject the depth pixels inside the box, keep the nearest surface
   that is above the floor, and measure it. Accurate, and tells us whether the
   object is really on the floor.
2. Floor plane: if depth is missing (clear plastic, shiny cans, too close),
   cast rays through the box edges and intersect them with the floor. This
   assumes the object rests on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics of the (raw, unrotated) color image."""

    fx: float
    fy: float
    cx: float
    cy: float

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(N, 3) rays in the optical frame, scaled so that z == 1 (so point = depth * ray)."""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)


@dataclass(frozen=True)
class Transform:
    """Rigid transform taking camera-frame points into the base frame."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def points(self, p: np.ndarray) -> np.ndarray:
        return p @ self.rotation.T + self.translation

    def directions(self, d: np.ndarray) -> np.ndarray:
        return d @ self.rotation.T


@dataclass(frozen=True)
class Box:
    """Detection box in raw image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Localization:
    position: np.ndarray  # (3,) object center in the base frame
    width: float  # largest horizontal extent, m
    height: float  # top of the object above the floor, m
    bottom: float  # lowest visible point above the floor, m (0 for the floor-plane estimate)
    depth_valid: bool  # False if this came from the floor-plane fallback


@dataclass(frozen=True)
class LocalizerParams:
    min_depth: float = 0.2  # m; D435i depth below this is unreliable
    max_depth: float = 4.0  # m
    floor_margin: float = 0.01  # m; points lower than this are floor, not object
    max_object_depth: float = 0.30  # m; never keep points further than this behind the nearest surface
    surface_gap: float = 0.03  # m; a jump in range this big separates the object from what's behind it
    box_shrink: float = 0.1  # ignore this fraction of the box on each side (mostly background)
    stride: int = 2  # sample every n-th pixel
    min_points: int = 20


def localize(box: Box, depth: np.ndarray, camera: CameraModel, camera_to_base: Transform,
             params: LocalizerParams = LocalizerParams()) -> Localization | None:
    """Locate the object in `box`. `depth` is in meters, aligned to the color image (0 = no data).

    Returns None if neither strategy works (e.g. the box is entirely above the horizon).
    """
    return (_from_depth(box, depth, camera, camera_to_base, params)
            or _from_floor_plane(box, camera, camera_to_base))


def _from_depth(box: Box, depth: np.ndarray, camera: CameraModel, camera_to_base: Transform,
                params: LocalizerParams) -> Localization | None:
    shrink_x = (box.x2 - box.x1) * params.box_shrink
    shrink_y = (box.y2 - box.y1) * params.box_shrink
    height_px, width_px = depth.shape
    x1, x2 = max(0, int(box.x1 + shrink_x)), min(width_px, int(np.ceil(box.x2 - shrink_x)))
    y1, y2 = max(0, int(box.y1 + shrink_y)), min(height_px, int(np.ceil(box.y2 - shrink_y)))
    if x2 <= x1 or y2 <= y1:
        return None

    v, u = np.mgrid[y1:y2:params.stride, x1:x2:params.stride]
    d = depth[v, u]
    valid = np.isfinite(d) & (d > params.min_depth) & (d < params.max_depth)
    if valid.sum() < params.min_points:
        return None
    u, v, d = u[valid], v[valid], d[valid]
    points = camera_to_base.points(camera.rays(u, v) * d[:, None])

    # Drop the floor around the object, then keep only the nearest surface so a
    # wall or furniture behind the object doesn't pull the estimate away.
    points = points[points[:, 2] > params.floor_margin]
    if len(points) < params.min_points:
        return None
    ranges = np.linalg.norm(points[:, :2] - camera_to_base.translation[:2], axis=1)
    points = points[_nearest_surface(ranges, params)]

    center_xy, width = _horizontal_extent(points[:, :2])
    top = float(np.percentile(points[:, 2], 95))
    bottom = float(np.percentile(points[:, 2], 5))
    return Localization(
        position=np.array([center_xy[0], center_xy[1], top / 2]),
        width=width, height=top, bottom=bottom, depth_valid=True,
    )


def _nearest_surface(ranges: np.ndarray, params: LocalizerParams) -> np.ndarray:
    """Indices of the points on the nearest continuous surface, by horizontal range from the camera.

    Horizontal range, not optical depth: with the camera tilted down, a wall
    *behind* an object is often closer in optical depth than the object itself.
    Sorted by range, an object's points form a dense run and anything behind it
    starts after a gap. The first `min_points` are skipped when looking for a
    gap, so a few stray noisy pixels in front can't cut the object off.
    """
    order = np.argsort(ranges)
    sorted_ranges = ranges[order]
    start = min(params.min_points, len(sorted_ranges) - 1)
    gaps = np.nonzero(np.diff(sorted_ranges[start:]) > params.surface_gap)[0]
    end = start + gaps[0] + 1 if gaps.size else len(sorted_ranges)
    keep = order[:end]
    return keep[ranges[keep] <= sorted_ranges[start] + params.max_object_depth]


def _horizontal_extent(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Center and largest extent of a robust bounding box aligned to the points' principal axes.

    The box center, not the median, is used: the visible points lean toward the camera
    (front face), and the box center counteracts that whenever the top face is visible.
    """
    mean = xy.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((xy - mean).T))
    projected = (xy - mean) @ axes
    low, high = np.percentile(projected, 2, axis=0), np.percentile(projected, 98, axis=0)
    center = mean + axes @ ((low + high) / 2)
    return center, float((high - low).max())


def _floor_hits(pixels: np.ndarray, camera: CameraModel, camera_to_base: Transform) -> np.ndarray:
    """Where rays through (N, 2) pixels hit the floor, in the base frame; NaN rows if they don't."""
    origin = camera_to_base.translation
    directions = camera_to_base.directions(camera.rays(pixels[:, 0], pixels[:, 1]))
    hits = np.full((len(pixels), 3), np.nan)
    downward = directions[:, 2] < -1e-9
    t = -origin[2] / directions[downward, 2]
    hits[downward] = origin + t[:, None] * directions[downward]
    return hits


def _from_floor_plane(box: Box, camera: CameraModel, camera_to_base: Transform) -> Localization | None:
    """Estimate from the box outline alone, assuming the object rests on the floor.

    Works in any image orientation: the edge whose midpoint ray hits the floor
    closest to the robot is where the object touches the floor.
    """
    corners = np.array([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # top, right, bottom, left (in raw pixels)
    midpoints = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
    midpoint_hits = _floor_hits(midpoints, camera, camera_to_base)

    origin = camera_to_base.translation
    ranges = np.linalg.norm(midpoint_hits[:, :2] - origin[:2], axis=1)
    if np.all(np.isnan(ranges)):
        return None
    near = int(np.nanargmin(ranges))
    contact = midpoint_hits[near]

    a, b = edges[near]
    corner_hits = _floor_hits(corners[[a, b]], camera, camera_to_base)
    width = float(np.linalg.norm(corner_hits[0, :2] - corner_hits[1, :2]))
    if not np.isfinite(width):
        return None

    # Height: follow the ray through the opposite edge to the contact point's range.
    far_direction = camera_to_base.directions(camera.rays(*midpoints[(near + 2) % 4]))
    contact_range = ranges[near]
    horizontal = np.linalg.norm(far_direction[:2])
    height = max(0.0, float(origin[2] + contact_range / horizontal * far_direction[2])) if horizontal > 0 else 0.0

    # The contact point is the near edge; the center is about half a width further away.
    away = (contact[:2] - origin[:2]) / contact_range
    center_xy = contact[:2] + away * width / 2
    return Localization(
        position=np.array([center_xy[0], center_xy[1], height / 2]),
        width=width, height=height, bottom=0.0, depth_valid=False,
    )
