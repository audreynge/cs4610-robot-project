"""Unit tests for geometry.py. Run with:  python3 -m pytest test/  (no ROS needed)."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clean_sweep_manipulation import geometry as g  # noqa: E402


def close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


def test_wrap_angle():
    assert close(g.wrap_angle(0.0), 0.0)
    assert close(g.wrap_angle(3 * math.pi / 2), -math.pi / 2)
    assert close(g.wrap_angle(-3 * math.pi / 2), math.pi / 2)


def test_quat_identity_and_z90():
    assert close(g.quat_to_matrix(0, 0, 0, 1), np.eye(3))
    s = math.sin(math.pi / 4)
    Rz90 = g.quat_to_matrix(0, 0, s, s)
    assert close(Rz90 @ [1, 0, 0], [0, 1, 0])


def test_transform_point():
    s = math.sin(math.pi / 4)
    p = g.transform_point((1, 2, 3), (0, 0, s, s), (1, 0, 0))
    assert close(p, [1, 3, 3])


def test_rotation_puts_object_on_arm_side():
    for obj in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.5), (0.3, -2.0), (2.0, 2.0)]:
        theta = g.base_rotation_to_put_object_on_arm_side(obj)
        p = g.point_after_base_rotation((obj[0], obj[1], 0.0), theta)
        # After rotating, object should be straight out on -y (x ~ 0, y < 0).
        assert close(p[0], 0.0), (obj, p)
        assert p[1] < 0, (obj, p)
        assert close(-p[1], math.hypot(*obj))


def test_object_ahead_needs_plus_90():
    theta = g.base_rotation_to_put_object_on_arm_side((1.0, 0.0))
    assert close(theta, math.pi / 2)


def test_plan_reach_basic():
    # Object 0.6 m to the right, 0.1 m ahead of grasp center, on the floor.
    plan = g.plan_reach(obj_base=(0.1, -0.6, 0.02),
                        grasp_center_base=(0.0, -0.3, 0.50),
                        current_arm_ext=0.05, current_lift=0.60, z_offset=0.0)
    assert close(plan["drive_x"], 0.1)
    assert close(plan["arm_ext"], 0.35)   # 0.05 + (-0.3 - -0.6)
    assert close(plan["lift"], 0.12)      # 0.60 + (0.02 - 0.50)
    assert plan["clamped"] is False


def test_plan_reach_clamps():
    plan = g.plan_reach(obj_base=(0.0, -2.0, -0.5),
                        grasp_center_base=(0.0, -0.3, 0.2),
                        current_arm_ext=0.0, current_lift=0.3)
    assert close(plan["arm_ext"], 0.52)
    assert close(plan["lift"], 0.0)
    assert plan["clamped"] is True


def test_gripper_width_and_holding():
    assert close(g.gripper_open_width(0.05, 0.02, 0.03, 0.09), 0.07)
    assert close(g.gripper_open_width(0.20, 0.02, 0.03, 0.09), 0.09)
    assert g.is_holding_object(0.10, empty_closed_pos=-0.05, tolerance=0.05)
    assert not g.is_holding_object(-0.04, empty_closed_pos=-0.05, tolerance=0.05)
    # Effort check: position looks held but motor isn't pushing -> not holding.
    assert not g.is_holding_object(0.10, -0.05, 0.05, finger_effort=2.0, min_effort=10.0)
    assert g.is_holding_object(0.10, -0.05, 0.05, finger_effort=-15.0, min_effort=10.0)


def test_check_reach():
    grasp = (0.0, -0.25, 0.5)
    ok = g.check_reach((0.01, -0.5, 0.0), grasp, x_tolerance=0.03, max_side_distance=0.52)
    assert ok["ok"]
    behind = g.check_reach((0.10, -0.5, 0.0), grasp, 0.03, 0.52)
    assert not behind["ok"] and close(behind["dx"], 0.10)
    far = g.check_reach((0.0, -0.9, 0.0), grasp, 0.03, 0.52)
    assert not far["ok"] and "beyond" in far["reason"]
    wrong_side = g.check_reach((0.0, 0.4, 0.0), grasp, 0.03, 0.52)
    assert not wrong_side["ok"] and "wrong side" in wrong_side["reason"]


def test_grasp_height():
    assert close(g.grasp_height(0.0, 0.12), 0.06)          # halfway up a can
    assert close(g.grasp_height(0.0, 0.01), 0.015)         # flat paper: min height
    assert close(g.grasp_height(0.02, 0.10, offset=0.01), 0.08)
