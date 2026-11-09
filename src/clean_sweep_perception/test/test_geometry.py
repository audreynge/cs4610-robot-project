"""Geometry tests against a synthetic camera that renders exact depth for boxes on a floor."""

import numpy as np
import pytest

from clean_sweep_perception.checks import SIZE_LIMITS, rejection_reason
from clean_sweep_perception.geometry import Box, CameraModel, Localization, Transform, localize
from clean_sweep_perception.labels import LITTER_LABELS

WIDTH, HEIGHT = 640, 480
CAMERA = CameraModel(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def head_camera(height=1.2, pitch_deg=45.0) -> Transform:
    """Camera above the base origin, looking along +x and pitched down."""
    s, c = np.sin(np.radians(pitch_deg)), np.cos(np.radians(pitch_deg))
    z_axis = np.array([c, 0.0, -s])  # optical axis
    x_axis = np.array([0.0, -1.0, 0.0])  # image right = robot's right
    y_axis = np.cross(z_axis, x_axis)  # image down
    return Transform(np.column_stack([x_axis, y_axis, z_axis]), np.array([0.0, 0.0, height]))


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


def test_depth_estimate_finds_center_and_size():
    pose = head_camera()
    can = can_at(1.1, 0.15)
    result = localize(project_box(pose, *can), render_depth(pose, [can]), CAMERA, pose)

    assert result.depth_valid
    np.testing.assert_allclose(result.position[:2], [1.1, 0.15], atol=0.015)
    assert 0.06 <= result.width <= 0.066 * np.sqrt(2)  # between face-on and diagonal
    assert result.height == pytest.approx(0.12, abs=0.015)
    assert result.bottom < 0.05


def test_ignores_wall_behind_object():
    pose = head_camera()
    can = can_at(1.1, 0.0)
    wall = ((1.2, -2.0, 0.0), (1.3, 2.0, 2.0))  # just behind the can
    box = project_box(pose, *can)
    box = Box(box.x1 - 15, box.y1 - 40, box.x2 + 15, box.y2)  # loose box that includes the wall
    result = localize(box, render_depth(pose, [can, wall]), CAMERA, pose)

    assert result.depth_valid
    np.testing.assert_allclose(result.position[:2], [1.1, 0.0], atol=0.02)


def test_falls_back_to_floor_plane_without_depth():
    """Clear plastic: the RealSense returns no depth on the object."""
    pose = head_camera()
    can = can_at(1.1, 0.15)
    no_depth = np.zeros((HEIGHT, WIDTH))
    result = localize(project_box(pose, *can), no_depth, CAMERA, pose)

    assert not result.depth_valid
    np.testing.assert_allclose(result.position[:2], [1.1, 0.15], atol=0.04)
    assert 0.04 < result.width < 0.12
    assert 0.05 < result.height < 0.25


def test_floor_plane_works_with_rotated_camera():
    """Stretch's head camera is mounted sideways; the fallback must not assume 'bottom edge'."""
    pose = head_camera()
    roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90 deg about optical axis
    rotated = Transform(pose.rotation @ roll, pose.translation)
    can = can_at(1.1, 0.0)
    result = localize(project_box(rotated, *can), np.zeros((HEIGHT, WIDTH)), CAMERA, rotated)

    np.testing.assert_allclose(result.position[:2], [1.1, 0.0], atol=0.04)


def test_box_above_horizon_cannot_be_placed():
    pose = head_camera(pitch_deg=0.0)  # looking straight ahead
    box = Box(300, 100, 340, 140)  # above the image center: rays go up
    assert localize(box, np.zeros((HEIGHT, WIDTH)), CAMERA, pose) is None


def test_object_on_table_is_rejected():
    pose = head_camera()
    table = ((0.9, -0.5, 0.0), (1.5, 0.5, 0.45))
    cup = can_at(1.05, 0.0, base_z=0.45)
    result = localize(project_box(pose, *cup), render_depth(pose, [table, cup]), CAMERA, pose)

    assert result.depth_valid
    assert result.bottom > 0.3
    assert "not on the floor" in rejection_reason(result, "cup")


def loc(width, height, bottom=0.0):
    return Localization(np.zeros(3), width, height, bottom, True)


def test_rejection_reasons():
    assert rejection_reason(loc(0.07, 0.12), "can") is None
    assert rejection_reason(loc(0.07, 0.12), None) is None
    assert "too big" in rejection_reason(loc(0.40, 0.12), "can")
    assert "too small" in rejection_reason(loc(0.01, 0.01), "paper")


def test_size_limits_cover_every_label():
    assert set(SIZE_LIMITS) == set(LITTER_LABELS)


@pytest.mark.parametrize("with_wall", [False, True])
def test_robust_to_realsense_like_noise(with_wall):
    """~1% depth noise plus a sprinkle of dropouts and stray pixels, as on a D435i at ~1.5 m."""
    rng = np.random.default_rng(0)
    pose = head_camera()
    can = can_at(1.1, 0.0)
    scene = [can, ((1.2, -2.0, 0.0), (1.3, 2.0, 2.0))] if with_wall else [can]
    depth = render_depth(pose, scene)
    depth *= 1 + rng.normal(0, 0.01, depth.shape)
    depth[rng.random(depth.shape) < 0.05] = 0.0  # dropouts
    stray = rng.random(depth.shape) < 0.01
    depth[stray] = rng.uniform(0.3, 3.0, stray.sum())  # flying pixels

    result = localize(project_box(pose, *can), depth, CAMERA, pose)
    assert result.depth_valid
    np.testing.assert_allclose(result.position[:2], [1.1, 0.0], atol=0.025)
    assert result.height == pytest.approx(0.12, abs=0.03)
    assert result.bottom < 0.05
