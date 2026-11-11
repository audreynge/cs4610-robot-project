"""Pure math helpers for the pick routine.

No ROS imports here on purpose, so this file can be unit-tested on any laptop
(see test/test_geometry.py).

Frame conventions (Stretch base_link):
    x = forward, y = left, z = up.
    Stretch's telescoping arm extends toward -y (the robot's right side).
    Positive base rotation = counter-clockwise when seen from above.
"""

import math

import numpy as np


def wrap_angle(a):
    """Wrap an angle in radians to the range [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_matrix(x, y, z, w):
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero-length quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def transform_point(translation, quaternion, point):
    """Apply a transform to a point: p_target = R * p_source + t.

    translation: (tx, ty, tz)  - source frame's origin, written in the target frame
    quaternion:  (x, y, z, w)  - source frame's orientation in the target frame
    point:       (px, py, pz)  - the point written in the source frame
    """
    R = quat_to_matrix(*quaternion)
    return R @ np.asarray(point, dtype=float) + np.asarray(translation, dtype=float)


def base_rotation_to_put_object_on_arm_side(obj_xy):
    """How far to rotate the base so the object ends up on the arm side (-y).

    In base_link the object is at angle atan2(y, x). We want it at -90 deg
    (straight out to the robot's right). Rotating the robot by +theta moves
    every object by -theta in the robot's frame, so
        theta = atan2(y, x) - (-pi/2) = atan2(y, x) + pi/2
    """
    x, y = obj_xy
    return wrap_angle(math.atan2(y, x) + math.pi / 2.0)


def point_after_base_rotation(point, theta):
    """Where a fixed world point appears in base_link after the robot spins by theta.

    The robot turns by +theta about its own z axis, so in the robot's frame
    the point turns by -theta. Height (z) is unchanged.
    """
    px, py, pz = point
    c, s = math.cos(-theta), math.sin(-theta)
    return np.array([c * px - s * py, s * px + c * py, pz])


def point_after_base_translation(point, dx):
    """Where a fixed world point appears in base_link after driving forward by dx."""
    px, py, pz = point
    return np.array([px - dx, py, pz])


def plan_reach(obj_base, grasp_center_base, current_arm_ext, current_lift,
               z_offset=0.0, arm_limits=(0.0, 0.52), lift_limits=(0.0, 1.1)):
    """Compute base drive, arm extension and lift height so the grasp center
    reaches the object.

    Assumes the base has already been rotated so the object is on the arm side.

    obj_base:          object position in base_link (x, y, z)
    grasp_center_base: current grasp-center position in base_link (x, y, z)
    current_arm_ext:   current wrist_extension (m)
    current_lift:      current joint_lift (m)
    z_offset:          stop this far above the object's point (m), e.g. half its height

    Returns dict with:
        drive_x  - how far to translate the base forward (m)
        arm_ext  - target wrist_extension (m), clamped
        lift     - target joint_lift (m), clamped
        clamped  - True if a limit was hit (object may be out of reach)
    """
    ox, oy, oz = obj_base
    gx, gy, gz = grasp_center_base

    # Line the grasp center up with the object along x by driving the base.
    drive_x = ox - gx

    # Extending the arm moves the grasp center toward -y, so to go from gy to oy
    # we need to extend by (gy - oy).
    arm_target = current_arm_ext + (gy - oy)

    # Moving the lift moves the grasp center up/down by the same amount.
    lift_target = current_lift + ((oz + z_offset) - gz)

    arm_c = min(max(arm_target, arm_limits[0]), arm_limits[1])
    lift_c = min(max(lift_target, lift_limits[0]), lift_limits[1])
    clamped = (abs(arm_c - arm_target) > 1e-6) or (abs(lift_c - lift_target) > 1e-6)

    return {"drive_x": drive_x, "arm_ext": arm_c, "lift": lift_c, "clamped": clamped}


def gripper_open_width(object_width, margin, min_open, max_open):
    """Aperture to open to before grasping: object width plus a margin, clamped."""
    return min(max(object_width + margin, min_open), max_open)


def is_holding_object(finger_pos_after_close, empty_closed_pos, tolerance,
                      finger_effort=None, min_effort=0.0):
    """Grasp check from finger position and effort.

    Position: if the fingers stopped noticeably before where they stop with
    nothing in between, something is blocking them.
    Effort: when squeezing something, the gripper motor pushes harder than
    when closed on air. min_effort=0 turns the effort check off.

    Both checks must pass (position AND effort), which avoids false positives
    from a finger that stopped early for another reason.
    """
    pos_ok = (finger_pos_after_close - empty_closed_pos) > tolerance
    eff_ok = True
    if min_effort > 0.0 and finger_effort is not None:
        eff_ok = abs(finger_effort) >= min_effort
    return pos_ok and eff_ok


def check_reach(obj_base, grasp_center_base, x_tolerance,
                max_side_distance, min_side_distance=0.0):
    """Is the object beside the robot and within arm reach?

    Navigation is responsible for parking the robot so the object sits on the
    arm side (-y). This checks that, and if not, says how far the base should
    move so navigation (or the state machine) can fix it.

    Returns dict:
        ok          - True if we can pick without moving the base
        dx          - drive the base forward by this much to line up (m)
        side_dist   - how far out to the side the object is from the grasp center (m)
        reason      - "" if ok, otherwise a short explanation
    """
    ox, oy, _ = obj_base
    gx, gy, _ = grasp_center_base
    dx = ox - gx
    side = gy - oy          # positive = object is further out on the arm side
    if oy > 0:
        return {"ok": False, "dx": dx, "side_dist": side,
                "reason": "object is on the wrong side of the robot (needs a turn)"}
    if side < min_side_distance:
        return {"ok": False, "dx": dx, "side_dist": side,
                "reason": "object is too close to the robot (behind the gripper)"}
    if side > max_side_distance:
        return {"ok": False, "dx": dx, "side_dist": side,
                "reason": "object is beyond full arm extension"}
    if abs(dx) > x_tolerance:
        return {"ok": False, "dx": dx, "side_dist": side,
                "reason": "object is not lined up with the arm (drive the base by dx)"}
    return {"ok": True, "dx": dx, "side_dist": side, "reason": ""}


def grasp_height(floor_z, object_height, fraction=0.5, min_height=0.015, offset=0.0):
    """Height (same frame as floor_z) the grasp center should aim at.

    Perception reports each object's center ON the floor plus its height, so we aim
    `fraction` of the way up the object, never closer to the floor than min_height.
    """
    return floor_z + max(object_height * fraction, min_height) + offset
