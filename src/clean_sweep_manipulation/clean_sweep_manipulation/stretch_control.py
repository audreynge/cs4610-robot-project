"""Manipulation logic: lift, arm, wrist, gripper, pick and bin drop.

No ROS imports: every robot interaction goes through an `io` object
(robot_io.RosStretchIO on the robot/sim, a fake in the tests), so the whole pick
sequence is covered by `./run.sh --test`.

The `io` object must provide:
    move(pose: dict[str, float], efforts: dict[str, float] | None = None) -> bool   (blocking)
    joint(name) -> (position, velocity, effort)
    joint_names() -> list[str]
    lookup(target_frame, source_frame) -> ((tx, ty, tz), (qx, qy, qz, qw)) or None
    switch_mode(name) -> bool          ("position" | "navigation")
    stow() -> bool
    sleep(seconds)
    log(level, message)                ("info" | "warn" | "error")
"""

import math

from clean_sweep_manipulation import config as C
from clean_sweep_manipulation import geometry as G


class StretchControl:

    # Result codes the state machine branches on.
    SUCCESS = "SUCCESS"                    # object in the gripper, arm retracted (out of the camera's view)
    GRASP_FAILED = "GRASP_FAILED"          # tried 1 + retries times; nothing in the gripper
    NEEDS_REPOSITION = "NEEDS_REPOSITION"  # not beside the arm / out of reach: see result["reach"]
    TOO_LARGE = "TOO_LARGE"                # wider than the gripper opens: skip it
    ERROR = "ERROR"                        # e.g. no TF to the object's frame: see result["message"]

    def __init__(self, io, grasp_mode=C.GRASP_MODE):
        self.io = io
        self.grasp_mode = grasp_mode

    # ------------------------------------------------------------ logging
    def _info(self, msg):
        self.io.log("info", msg)

    def _warn(self, msg):
        self.io.log("warn", msg)

    # ------------------------------------------------------------- modes
    def position_mode(self):
        return self.io.switch_mode("position")

    def navigation_mode(self):
        return self.io.switch_mode("navigation")

    # ------------------------------------------------------- reading state
    def joint(self, name):
        return self.io.joint(name)

    def lift_height(self):
        return self.io.joint(C.LIFT)[0]

    def arm_extension(self):
        """Total telescoping arm extension (m)."""
        if C.ARM in self.io.joint_names():
            return self.io.joint(C.ARM)[0]
        return sum(self.io.joint(n)[0] for n in C.ARM_SEGMENTS)

    def gripper_finger(self):
        """(position, effort) of the left finger."""
        pos, _, eff = self.io.joint(C.GRIPPER_FINGER)
        return pos, eff

    def grasp_center_in_base(self):
        """Current (x, y, z) of the point between the fingertips, in base_link."""
        tf = self.io.lookup(C.BASE_FRAME, C.GRASP_FRAME)
        if tf is None:
            raise RuntimeError(f"no TF {C.BASE_FRAME} -> {C.GRASP_FRAME}")
        return tuple(tf[0])

    def to_base_frame(self, point, source_frame):
        """Convert a point (x, y, z) in source_frame (e.g. "map") to base_link."""
        if source_frame in ("", C.BASE_FRAME):
            return tuple(point)
        tf = self.io.lookup(C.BASE_FRAME, source_frame)
        if tf is None:
            raise RuntimeError(f"no TF {C.BASE_FRAME} -> {source_frame}")
        return tuple(G.transform_point(tf[0], tf[1], point))

    # ---------------------------------------------------- single motions
    # All blocking. Each returns True if the driver reported success.
    def set_lift(self, height_m):
        return self.io.move({C.LIFT: _clamp(height_m, C.LIFT_LIMITS)})

    def lower_lift_until_contact(self, height_m, effort=C.LIFT_CONTACT_EFFORT):
        """Lower the lift, stopping early if it presses on something (floor or object)."""
        return self.io.move({C.LIFT: _clamp(height_m, C.LIFT_LIMITS)}, efforts={C.LIFT: effort})

    def set_arm(self, extension_m):
        return self.io.move({C.ARM: _clamp(extension_m, C.ARM_LIMITS)})

    def retract_arm(self):
        return self.set_arm(C.ARM_LIMITS[0])

    def wrist_grasp_pose(self):
        """Standard gripper: straight out. Dex wrist: pointing down."""
        return self.io.move(dict(C.WRIST_GRASP_POSE))

    def set_gripper(self, aperture):
        return self.io.move({C.GRIPPER_CMD: aperture})

    def open_gripper(self, width=None):
        """Open fully, or to `width` (m), clamped to what the gripper can do."""
        g = C.GRIPPER
        a = g["open"] if width is None else _clamp(width, (g["min_open"], g["max_open"]))
        return self.set_gripper(a)

    def close_gripper(self):
        return self.set_gripper(C.GRIPPER["closed"])

    def set_head(self, pan=None, tilt=None):
        pose = {}
        if pan is not None:
            pose[C.HEAD_PAN] = pan
        if tilt is not None:
            pose[C.HEAD_TILT] = tilt
        return self.io.move(pose) if pose else True

    def rotate_base(self, angle_rad):
        """Turn in place (positive = counter-clockwise). Position mode only."""
        return self.io.move({C.BASE_ROTATE: angle_rad})

    def drive_base(self, distance_m):
        """Drive straight (negative = backward). Position mode only."""
        return self.io.move({C.BASE_TRANSLATE: distance_m})

    def stow(self):
        return self.io.stow()

    def travel_pose(self):
        """Arm in, lift at a safe height, wrist in grasp pose: before driving or reaching."""
        self.retract_arm()
        self.set_lift(C.SAFE_LIFT)
        self.wrist_grasp_pose()

    # ------------------------------------------------------- grasp check
    def is_holding(self):
        """True if the fingers stopped early on something AND (if enabled) are squeezing."""
        pos, eff = self.gripper_finger()
        g = C.GRIPPER
        holding = G.is_holding_object(pos, g["empty_closed_finger"], g["holding_tolerance"],
                                      finger_effort=eff, min_effort=g["min_holding_effort"])
        self._info(f"grasp check: finger={pos:.3f} effort={eff:.1f} holding={holding}")
        return holding

    # ---------------------------------------------------------- picking
    def check_object_reachable(self, obj_base):
        """Is the object beside the arm and within reach? (geometry.check_reach dict)"""
        grasp = self.grasp_center_in_base()
        max_side = C.ARM_LIMITS[1] - self.arm_extension()
        return G.check_reach(obj_base, grasp, C.REACH_X_TOLERANCE,
                             max_side_distance=max_side, min_side_distance=C.MIN_SIDE_DISTANCE)

    def align_base_to_object(self, obj_base):
        """Turn and drive so the object is beside the arm; returns its new base_link point.

        NAVIGATION owns the final approach, so picks only call this with
        align_base=True (for testing manipulation on its own).
        """
        theta = G.base_rotation_to_put_object_on_arm_side(obj_base[:2])
        self._info(f"align: rotating base {math.degrees(theta):.1f} deg")
        self.rotate_base(theta)
        obj = G.point_after_base_rotation(obj_base, theta)
        dx = obj[0] - self.grasp_center_in_base()[0]
        if abs(dx) > 0.005:
            self.drive_base(dx)
            obj = G.point_after_base_translation(obj, dx)
        return tuple(obj)

    def pick_litter(self, litter, retries=C.DEFAULT_RETRIES, align_base=False):
        """Pick a clean_sweep_interfaces/LitterObject (use the one litter/refine returns).

        Uses its position (object center on the floor, any frame with TF to
        base_link, normally "map"), width and height.
        """
        frame = litter.position.header.frame_id
        p = litter.position.point
        try:
            floor_base = self.to_base_frame((p.x, p.y, p.z), frame)
        except RuntimeError as e:
            self._warn(f"pick: {e}")
            return {"result": self.ERROR, "attempts": 0, "reach": None, "message": str(e)}
        return self.pick_object_at(floor_base, object_width=float(litter.width),
                                   object_height=float(litter.height),
                                   retries=retries, align_base=align_base)

    def pick_object_at(self, floor_point_base, object_width=0.05, object_height=0.05,
                       retries=C.DEFAULT_RETRIES, align_base=False):
        """Pick an object whose center ON THE FLOOR is floor_point_base (base_link).

        Assumes navigation parked the robot with the object beside it on the arm side
        (-y), within reach; align_base=True lets this function turn/drive itself instead.

        Side grasp (standard gripper): open, lower to the grasp height, extend around
        the object, close. Top grasp (dex wrist): open, extend over it, lower until
        contact, close. Then check the grasp (finger position and effort), lift, check
        again (catches slips), retract. On failure, open wider and retry.

        Returns {"result": <code>, "attempts": n, "reach": dict | None, "message": str}.
        """
        g = C.GRIPPER
        if object_width > g["max_open"]:
            msg = f"object width {object_width:.3f} m > gripper max {g['max_open']:.3f} m"
            self._warn(f"pick: {msg}, skipping")
            return {"result": self.TOO_LARGE, "attempts": 0, "reach": None, "message": msg}

        self.position_mode()
        self.travel_pose()

        target_z = G.grasp_height(floor_point_base[2], object_height, C.GRASP_HEIGHT_FRACTION,
                                  C.GRASP_MIN_HEIGHT, C.GRASP_Z_OFFSET)
        obj = (floor_point_base[0], floor_point_base[1], target_z)
        if align_base:
            obj = self.align_base_to_object(obj)

        reach = self.check_object_reachable(obj)
        if not reach["ok"]:
            self._warn(f"pick: cannot reach: {reach['reason']} (dx={reach['dx']:.3f})")
            return {"result": self.NEEDS_REPOSITION, "attempts": 0, "reach": reach,
                    "message": reach["reason"]}

        open_width = G.gripper_open_width(object_width, C.GRASP_WIDTH_MARGIN,
                                          g["min_open"], g["max_open"])
        attempts = retries + 1
        for attempt in range(1, attempts + 1):
            plan = G.plan_reach(obj, self.grasp_center_in_base(), self.arm_extension(),
                                self.lift_height(), z_offset=0.0,
                                arm_limits=C.ARM_LIMITS, lift_limits=C.LIFT_LIMITS)
            self._info(f"pick attempt {attempt}/{attempts} ({self.grasp_mode}): {plan}")
            if plan["clamped"]:
                self._warn("pick: a joint limit was hit; the grasp may miss")

            self.open_gripper(open_width)
            if self.grasp_mode == "side":
                # Lower beside the object (arm still in), then reach around it.
                self.lower_lift_until_contact(plan["lift"])
                self.set_arm(plan["arm_ext"])
            else:
                # Reach over the object, then come down onto it.
                self.set_arm(plan["arm_ext"])
                self.lower_lift_until_contact(plan["lift"])

            self.close_gripper()
            self.io.sleep(C.GRASP_SETTLE_S)
            holding = self.is_holding()

            # Raise and check again: a weak grasp slips here.
            self.set_lift(C.SAFE_LIFT)
            if holding and self.is_holding():
                self.retract_arm()
                return {"result": self.SUCCESS, "attempts": attempt, "reach": reach, "message": ""}

            self._warn(f"pick: grasp attempt {attempt} failed")
            self.open_gripper()
            self.retract_arm()   # back out before the next try (side grasp must re-lower beside it)
            open_width = min(open_width + 0.01, g["max_open"])   # misses are often too narrow

        return {"result": self.GRASP_FAILED, "attempts": attempts, "reach": reach,
                "message": "grasp failed on every attempt"}

    # ---------------------------------------------------------- dropping
    def drop_in_bin(self, lift=C.BIN_DROP_LIFT, arm=C.BIN_DROP_ARM):
        """Navigation parked with the bin on the arm side: raise above the rim,
        reach over, open, come back, then confirm the gripper is empty.

        The empty check closes the gripper after retracting: open fingers can't tell
        whether something is stuck between them, but closed fingers can.
        Returns True if the object was released.
        """
        self.position_mode()
        self.set_lift(lift)
        self.set_arm(arm)
        self.open_gripper()
        self.io.sleep(C.GRASP_SETTLE_S)
        self.retract_arm()
        self.close_gripper()
        self.io.sleep(C.GRASP_SETTLE_S)
        released = not self.is_holding()
        if not released:
            self._warn("drop: object still in the gripper (did not fall out)")
        self.set_lift(C.SAFE_LIFT)
        return released


def _clamp(v, limits):
    return min(max(v, limits[0]), limits[1])
