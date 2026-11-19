"""Runs the pick and drop logic against a fake Stretch (no ROS).

The fake is a tiny kinematic model:
    grasp center (base_link) = (0, -(GRASP_Y0 + arm), lift - GRASP_Z_DROP)
When the gripper closes, it holds the object only if the grasp center is within
2 cm of the object's grasp point, so a wrong reach plan fails the test.
"""

import math
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clean_sweep_manipulation import config as C  # noqa: E402
from clean_sweep_manipulation import geometry as G  # noqa: E402
from clean_sweep_manipulation.stretch_control import StretchControl  # noqa: E402

GRASP_Y0 = 0.25
GRASP_Z_DROP = 0.20
S45 = math.sin(math.pi / 4)


class FakeIO:
    def __init__(self, obj_base=None, slips=False, wrist_camera=False, false_hold=False,
                 soft=False):
        self.j = {"joint_lift": 0.5, "wrist_extension": 0.0, "joint_gripper_finger_left": 0.0}
        self.eff = 0.0
        self.obj = obj_base          # (x, y, z) of the object's grasp point in base_link
        self.slips = slips
        self.wrist_camera = wrist_camera  # sees the object while it's on the floor
        self.false_hold = false_hold      # fingers read "holding" without the object
        self.soft = soft                  # held, but barely loads the fingers (paper)
        self.held = False
        self.calls = []              # ("move", pose, efforts) | ("mode", name)
        # map frame pose in base_link: robot at (1, 2) in map, facing +90 deg.
        self.map_tf = ((-2.0, 1.0, 0.0), (0.0, 0.0, -S45, S45))

    # --- io interface
    def joint_names(self):
        return list(self.j)

    def joint(self, name):
        return self.j[name], 0.0, (self.eff if name == "joint_gripper_finger_left" else 0.0)

    def grasp_center(self):
        return (0.0, -(GRASP_Y0 + self.j["wrist_extension"]), self.j["joint_lift"] - GRASP_Z_DROP)

    def lookup(self, target, source):
        if source == "link_grasp_center":
            return self.grasp_center(), (0, 0, 0, 1)
        if source == "map":
            return self.map_tf
        return None

    def move(self, pose, efforts=None):
        self.calls.append(("move", dict(pose), efforts))
        for name, val in pose.items():
            if name == "gripper_aperture":
                if val < 0:   # closing
                    g = self.grasp_center()
                    near = self.held or (self.obj is not None and math.dist(g, self.obj) < 0.02)
                    self.held = near
                    near = near or self.false_hold
                    self.j["joint_gripper_finger_left"] = 0.12 if near else -0.05
                    self.eff = -20.0 if near and not self.soft else -1.0
                else:
                    self.held = False
                    self.j["joint_gripper_finger_left"] = val
                    self.eff = 0.0
            elif name == "translate_mobile_base":
                if self.obj is not None:
                    self.obj = tuple(G.point_after_base_translation(self.obj, val))
            elif name == "rotate_mobile_base":
                if self.obj is not None:
                    self.obj = tuple(G.point_after_base_rotation(self.obj, val))
            else:
                self.j[name] = val
        if self.held and self.slips and pose.get("joint_lift") == C.SAFE_LIFT:
            self.held = False
            self.j["joint_gripper_finger_left"], self.eff = -0.05, -1.0
        return True

    def wrist_detections(self, timeout):
        if not self.wrist_camera:
            return None
        if self.obj is None:
            return []
        if self.held:   # up in the gripper, right under the camera: measures tall
            g = self.grasp_center()
            return [(g[0], g[1], g[2], g[2] + CAN_H / 2)]
        return [(self.obj[0], self.obj[1], self.obj[2], CAN_H)]

    def switch_mode(self, name):
        self.calls.append(("mode", name))
        return True

    def stow(self):
        return True

    def sleep(self, s):
        pass

    def log(self, level, msg):
        pass

    # --- helpers for asserts
    def moves(self, joint):
        return [c[1][joint] for c in self.calls if c[0] == "move" and joint in c[1]]


@pytest.fixture(autouse=True)
def calibrated(monkeypatch):
    monkeypatch.setitem(C.GRIPPER, "empty_closed_finger", -0.05)
    monkeypatch.setitem(C.GRIPPER, "holding_tolerance", 0.05)
    monkeypatch.setitem(C.GRIPPER, "min_holding_effort", 10.0)


CAN_FLOOR = (0.0, -0.55, 0.0)   # can's center on the floor, beside the arm
CAN_W, CAN_H = 0.066, 0.12
CAN_GRASP = (0.0, -0.55, 0.06)  # halfway up


@pytest.mark.parametrize("mode", ["side", "top"])
def test_pick_succeeds_and_reaches_the_right_spot(mode):
    io = FakeIO(obj_base=CAN_GRASP)
    out = StretchControl(io, grasp_mode=mode).pick_object_at(CAN_FLOOR, CAN_W, CAN_H)
    assert out["result"] == StretchControl.SUCCESS, out
    assert out["attempts"] == 1
    assert io.j["wrist_extension"] == 0.0        # retracted (out of the camera's view)
    assert io.moves("gripper_aperture")[0] == pytest.approx(C.GRIPPER["open"])   # fully open
    assert not io.moves("rotate_mobile_base")    # navigation owns the approach


def test_side_grasp_lowers_before_extending_and_top_grasp_the_other_way():
    for mode, first in (("side", "lift"), ("top", "arm")):
        io = FakeIO(obj_base=CAN_GRASP)
        StretchControl(io, grasp_mode=mode).pick_object_at(CAN_FLOOR, CAN_W, CAN_H)
        seq = []
        for c in io.calls:
            if c[0] != "move":
                continue
            if c[2]:                                      # the contact-limited lift lowering
                seq.append("lift")
            elif c[1].get("wrist_extension", 0) > 0:     # reaching out
                seq.append("arm")
        assert seq[0] == first, (mode, seq)


def test_lowering_uses_a_contact_threshold():
    io = FakeIO(obj_base=CAN_GRASP)
    StretchControl(io, grasp_mode="side").pick_object_at(CAN_FLOOR, CAN_W, CAN_H)
    lowered = [c for c in io.calls if c[0] == "move" and c[2]]
    assert lowered and lowered[0][2] == {"joint_lift": C.LIFT_CONTACT_EFFORT}


def test_missed_grasp_retries_then_gives_up():
    io = FakeIO(obj_base=None)                   # nothing there to grab
    out = StretchControl(io, grasp_mode="side").pick_object_at(CAN_FLOOR, 0.04, CAN_H, retries=2)
    assert out["result"] == StretchControl.GRASP_FAILED
    assert out["attempts"] == 3
    opens = [a for a in io.moves("gripper_aperture") if a > 0]
    assert opens and all(a == pytest.approx(C.GRIPPER["open"]) for a in opens)


def test_rechecks_after_the_gripper_backs_off():
    # Real gripper: stall protection drops the effort while lifting, though the object
    # is still held. The pick must squeeze again before the second check, not drop it.
    io = FakeIO(obj_base=CAN_GRASP)
    io_move = io.move

    def move(pose, efforts=None):
        ok = io_move(pose, efforts)
        if io.held and pose.get("joint_lift") == C.SAFE_LIFT:
            io.eff = -7.0                      # backed off, still holding
        return ok
    io.move = move
    out = StretchControl(io, grasp_mode="top").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.SUCCESS, out


def test_slip_on_lift_counts_as_failure():
    io = FakeIO(obj_base=CAN_GRASP, slips=True)
    out = StretchControl(io, grasp_mode="side").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.GRASP_FAILED


def test_too_large_skips_without_moving():
    io = FakeIO()
    out = StretchControl(io).pick_object_at(CAN_FLOOR, object_width=C.GRIPPER["max_open"] + 0.01,
                                         object_height=0.2)
    assert out["result"] == StretchControl.TOO_LARGE
    assert not [c for c in io.calls if c[0] == "move"]


def test_not_beside_arm_asks_navigation_to_reposition():
    io = FakeIO()
    out = StretchControl(io).pick_object_at((0.20, -0.55, 0.0), CAN_W, CAN_H)
    assert out["result"] == StretchControl.NEEDS_REPOSITION
    assert out["reach"]["dx"] == pytest.approx(0.20)
    out = StretchControl(io).pick_object_at((0.0, 0.5, 0.0), CAN_W, CAN_H)
    assert out["result"] == StretchControl.NEEDS_REPOSITION


def litter_msg(x, y, z, frame="map", width=CAN_W, height=CAN_H):
    """Duck-typed clean_sweep_interfaces/LitterObject."""
    pos = types.SimpleNamespace(header=types.SimpleNamespace(frame_id=frame),
                                point=types.SimpleNamespace(x=x, y=y, z=z))
    return types.SimpleNamespace(position=pos, width=width, height=height)


def test_pick_litter_converts_from_map_frame():
    # Robot at (1, 2) in map facing +90 deg; this map point is (0, -0.55, 0) in base_link.
    io = FakeIO(obj_base=CAN_GRASP)
    out = StretchControl(io, grasp_mode="side").pick_litter(litter_msg(1.55, 2.0, 0.0))
    assert out["result"] == StretchControl.SUCCESS, out


def test_pick_litter_without_tf_reports_error():
    io = FakeIO()
    out = StretchControl(io).pick_litter(litter_msg(1.0, 1.0, 0.0, frame="odom"))
    assert out["result"] == StretchControl.ERROR
    assert not [c for c in io.calls if c[0] == "move"]


def test_align_base_mode_turns_then_picks():
    # Object 0.5 m out on the right, 0.2 m ahead: within reach after one turn, no drive.
    io = FakeIO(obj_base=(0.2, -0.5, CAN_H / 2))
    out = StretchControl(io, grasp_mode="side").pick_object_at(
        (0.2, -0.5, 0.0), CAN_W, CAN_H, align_base=True, retries=0)
    assert out["result"] == StretchControl.SUCCESS, out
    assert len(io.moves("rotate_mobile_base")) == 1
    assert not io.moves("translate_mobile_base")


@pytest.mark.parametrize("floor_xy", [(0.8, 0.0), (0.3, -1.1), (-0.4, 0.9)])
def test_align_base_mode_drives_closer_when_out_of_reach(floor_xy):
    # Beyond 0.25 + 0.52 m after a turn alone: faces it, drives closer, turns, picks.
    io = FakeIO(obj_base=(*floor_xy, CAN_H / 2))
    out = StretchControl(io, grasp_mode="side").pick_object_at(
        (*floor_xy, 0.0), CAN_W, CAN_H, align_base=True, retries=0)
    assert out["result"] == StretchControl.SUCCESS, out
    assert io.moves("translate_mobile_base")[0] > 0


def test_drop_confirms_release_with_a_closed_gripper():
    io = FakeIO(obj_base=None)
    assert StretchControl(io).drop_in_bin() is True
    ap = io.moves("gripper_aperture")
    assert ap[0] > 0 and ap[-1] < 0              # opened over the bin, then closed to check
    assert io.j["wrist_extension"] == 0.0
    # Backs up first, then lifts, then reaches out.
    moves = [next(iter(c[1])) for c in io.calls if c[0] == "move"]
    assert moves[:3] == ["translate_mobile_base", "joint_lift", "wrist_extension"]
    assert io.calls[1][1]["translate_mobile_base"] == pytest.approx(-C.BIN_DROP_BACKUP)


def test_return_to_start_undoes_base_moves_in_reverse():
    io = FakeIO(obj_base=None)
    robot = StretchControl(io)
    robot.rotate_base(-0.08)
    robot.drive_base(0.02)
    robot.drop_in_bin()                          # backs up BIN_DROP_BACKUP
    io.calls.clear()
    assert robot.return_to_start() is True
    base = [c[1] for c in io.calls if c[0] == "move"
            and ("rotate_mobile_base" in c[1] or "translate_mobile_base" in c[1])]
    assert base == [{"translate_mobile_base": pytest.approx(C.BIN_DROP_BACKUP)},
                    {"translate_mobile_base": pytest.approx(-0.02)},
                    {"rotate_mobile_base": pytest.approx(0.08)}]
    assert io.j["wrist_extension"] == 0.0        # arm in before driving
    io.calls.clear()
    assert robot.return_to_start() is True and not io.calls   # nothing left to undo


def test_drop_detects_stuck_object():
    io = FakeIO(obj_base=None)
    robot = StretchControl(io)
    # Something still between the fingers when it closes to check:
    io.obj = io.grasp_center()
    real_move = io.move

    def move(pose, efforts=None):
        r = real_move(pose, efforts)
        io.obj = io.grasp_center()   # the object follows the gripper (stuck)
        return r
    io.move = move
    assert robot.drop_in_bin() is False


def test_wrist_camera_recenters_a_top_grasp():
    # The head camera's estimate is 4 cm off: the grasp misses without the wrist camera.
    actual = (CAN_GRASP[0] + 0.04, CAN_GRASP[1] - 0.03, CAN_GRASP[2])
    for wrist, expected in ((False, StretchControl.GRASP_FAILED), (True, StretchControl.SUCCESS)):
        io = FakeIO(obj_base=actual, wrist_camera=wrist)
        out = StretchControl(io, grasp_mode="top").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
        assert out["result"] == expected, (wrist, out)


def test_wrist_camera_catches_a_grasp_that_left_the_object_behind():
    # The finger effort says "holding", but the wrist camera still sees it on the floor.
    io = FakeIO(obj_base=(CAN_GRASP[0] + 0.04, *CAN_GRASP[1:]), wrist_camera=True, false_hold=True)
    robot = StretchControl(io, grasp_mode="top")
    robot.center_over_object = lambda obj: obj      # don't let it fix the aim
    out = robot.pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.GRASP_FAILED, out
    # Without the camera, the same false reading counts as a success.
    io = FakeIO(obj_base=(CAN_GRASP[0] + 0.04, *CAN_GRASP[1:]), false_hold=True)
    out = StretchControl(io, grasp_mode="top").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.SUCCESS


def test_wrist_camera_confirms_a_soft_grasp_the_fingers_barely_feel():
    # Paper: held, but the squeeze effort looks like an empty gripper.
    io = FakeIO(obj_base=CAN_GRASP, wrist_camera=True, soft=True)
    out = StretchControl(io, grasp_mode="top").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.SUCCESS, out
    # Without the camera, effort alone calls it a miss.
    io = FakeIO(obj_base=CAN_GRASP, soft=True)
    out = StretchControl(io, grasp_mode="top").pick_object_at(CAN_FLOOR, CAN_W, CAN_H, retries=0)
    assert out["result"] == StretchControl.GRASP_FAILED
