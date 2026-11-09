"""Geometry tests against a synthetic camera that renders exact depth for boxes on a floor."""

import numpy as np
import pytest

from clean_sweep_perception.checks import SIZE_LIMITS, rejection_reason
from clean_sweep_perception.geometry import Box, Localization, Transform, localize
from clean_sweep_perception.labels import LITTER_LABELS
from synthetic_scene import CAMERA, HEIGHT, WIDTH, can_at, head_camera, project_box, render_depth


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
