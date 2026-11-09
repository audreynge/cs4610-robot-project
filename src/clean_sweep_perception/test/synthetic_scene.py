"""A synthetic head camera that renders exact depth for boxes on a floor (test helper)."""

import numpy as np

from clean_sweep_perception.fake_camera import head_camera_pose
from clean_sweep_perception.geometry import Box, CameraModel, Transform

WIDTH, HEIGHT = 640, 480
CAMERA = CameraModel(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def head_camera(height=1.2, pitch_deg=45.0, rotation="none") -> Transform:
    return head_camera_pose(height, pitch_deg, rotation)


def render_depth(camera_to_base: Transform, cuboids) -> np.ndarray:
    """Depth (optical z) of the nearest surface per pixel: floor plus axis-aligned cuboids."""
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    directions = camera_to_base.directions(CAMERA.rays(u.ravel(), v.ravel()))
    origin = camera_to_base.translation

    with np.errstate(divide="ignore", invalid="ignore"):
        t_best = np.where(directions[:, 2] < 0, -origin[2] / directions[:, 2], np.inf)
        for low, high in cuboids:  # slab-method ray/box intersection
            t1 = (np.asarray(low) - origin) / directions
            t2 = (np.asarray(high) - origin) / directions
            t_near = np.nanmax(np.minimum(t1, t2), axis=1)
            t_far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (t_near <= t_far) & (t_near > 0)
            t_best = np.where(hit & (t_near < t_best), t_near, t_best)
    # Rays scaled to optical z == 1, so the ray parameter is the depth.
    return np.where(np.isfinite(t_best), t_best, 0.0).reshape(HEIGHT, WIDTH)


def project_box(camera_to_base: Transform, low, high) -> Box:
    """Tight image box around a cuboid's corners."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (low[2], high[2])])
    in_camera = (corners - camera_to_base.translation) @ camera_to_base.rotation
    u = CAMERA.fx * in_camera[:, 0] / in_camera[:, 2] + CAMERA.cx
    v = CAMERA.fy * in_camera[:, 1] / in_camera[:, 2] + CAMERA.cy
    return Box(u.min(), v.min(), u.max(), v.max())


def can_at(x, y, base_z=0.0, size=0.066, height=0.12):
    half = size / 2
    return (x - half, y - half, base_z), (x + half, y + half, base_z + height)
