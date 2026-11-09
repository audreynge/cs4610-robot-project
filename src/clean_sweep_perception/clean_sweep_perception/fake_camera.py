"""Synthesize a head-camera frame from an ordinary photo (no ROS).

Used to test the detector end to end without the robot: the photo is treated
as the upright view, turned into the sideways raw image the Stretch head camera
produces, and paired with the depth of a flat floor seen from the head.
"""

from __future__ import annotations

import numpy as np

from clean_sweep_perception.geometry import CameraModel, Transform
from clean_sweep_perception.image_rotation import ROTATIONS

D435_COLOR_HFOV_DEG = 69.0


def head_camera_pose(height: float = 1.2, pitch_deg: float = 45.0, rotation: str = "none") -> Transform:
    """Camera above the base origin looking along +x, pitched down, and rolled so that
    turning its raw image by `rotation` makes it upright."""
    s, c = np.sin(np.radians(pitch_deg)), np.cos(np.radians(pitch_deg))
    z_axis = np.array([c, 0.0, -s])  # optical axis
    x_axis = np.array([0.0, -1.0, 0.0])  # upright image right = robot's right
    y_axis = np.cross(z_axis, x_axis)  # upright image down
    upright = np.column_stack([x_axis, y_axis, z_axis])
    # A raw image that needs k clockwise turns comes from a camera rolled k*90 deg about its axis.
    k = ROTATIONS[rotation]
    cos_k, sin_k = round(np.cos(k * np.pi / 2)), round(np.sin(k * np.pi / 2))
    roll = np.array([[cos_k, -sin_k, 0], [sin_k, cos_k, 0], [0, 0, 1]], dtype=float)
    return Transform(upright @ roll, np.array([0.0, 0.0, height]))


def camera_for(shape: tuple[int, ...], hfov_deg: float = D435_COLOR_HFOV_DEG) -> CameraModel:
    """Pinhole model with the given field of view across the image's long side."""
    h, w = shape[:2]
    f = (max(w, h) / 2) / np.tan(np.radians(hfov_deg) / 2)
    return CameraModel(fx=f, fy=f, cx=w / 2, cy=h / 2)


def floor_depth(camera: CameraModel, pose: Transform, shape: tuple[int, ...]) -> np.ndarray:
    """Optical depth (m) of the floor plane z=0 at every pixel; 0 where the ray misses it."""
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    directions = pose.directions(camera.rays(u.ravel() + 0.5, v.ravel() + 0.5))
    with np.errstate(divide="ignore"):
        t = np.where(directions[:, 2] < 0, -pose.translation[2] / directions[:, 2], 0.0)
    return t.reshape(shape[:2]).astype(np.float32)


def fake_frame(upright_bgr: np.ndarray, rotation: str = "cw90", height: float = 1.2,
               pitch_deg: float = 45.0) -> tuple[np.ndarray, np.ndarray, CameraModel, Transform]:
    """(raw color, depth, intrinsics, camera->base) as if the head camera saw `upright_bgr`."""
    raw = np.ascontiguousarray(np.rot90(upright_bgr, k=ROTATIONS[rotation]))  # undo a cw turn
    camera = camera_for(raw.shape)
    pose = head_camera_pose(height, pitch_deg, rotation)
    return raw, floor_depth(camera, pose, raw.shape), camera, pose
