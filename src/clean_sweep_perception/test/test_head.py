import math

import numpy as np
import pytest

from clean_sweep_perception.head import HeadLimits, look_at_angles

HEAD = np.array([0.0, 0.0, 1.2])


def test_straight_ahead_and_down():
    pan, tilt = look_at_angles([1.2, 0.0, 0.0], HEAD)
    assert pan == pytest.approx(0.0)
    assert tilt == pytest.approx(-math.pi / 4)


def test_to_the_robots_right_is_negative_pan():
    pan, _ = look_at_angles([0.0, -1.0, 0.0], HEAD)
    assert pan == pytest.approx(-math.pi / 2)


def test_behind_the_robot_uses_the_reachable_equivalent_angle():
    # Straight behind is +pi, beyond pan_max (1.73); -pi is reachable.
    pan, _ = look_at_angles([-1.0, 0.0, 0.0], HEAD)
    assert pan == pytest.approx(-math.pi)


def test_pan_zero_heading_offsets_pan():
    pan, _ = look_at_angles([0.0, 1.0, 0.0], HEAD, pan_zero_heading=math.pi / 2)
    assert pan == pytest.approx(0.0)


def test_tilt_is_clamped_to_limits():
    _, tilt = look_at_angles([0.01, 0.0, 0.0], HEAD, limits=HeadLimits(tilt_min=-1.0))
    assert tilt == pytest.approx(-1.0)


def test_unreachable_pan_clamps_to_nearest_limit():
    limits = HeadLimits(pan_min=-1.0, pan_max=1.0)
    pan, _ = look_at_angles([-1.0, -0.1, 0.0], HEAD, limits=limits)  # about -3.04 rad
    assert pan == pytest.approx(-1.0)
