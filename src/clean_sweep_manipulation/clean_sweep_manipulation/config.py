"""Every tunable number for manipulation, in one place.

Values marked CALIBRATE are starting guesses: measure them with
`calibrate_gripper` and `demo_arm_gripper` in sim AND on the robot.

Two settings come from the environment so the same code runs everywhere:
    CLEAN_SWEEP_PLATFORM = real | sim     (gripper numbers differ between them)
    CLEAN_SWEEP_DEX_WRIST = 0 | 1         (Stretch 2 standard gripper = 0, the default;
                                           Stretch 3 / dex wrist = 1)
"""

import math
import os

PLATFORM = os.environ.get("CLEAN_SWEEP_PLATFORM", "real")
HAS_DEX_WRIST = os.environ.get("CLEAN_SWEEP_DEX_WRIST", "0") == "1"

# How we grasp:
#   "side": standard Stretch gripper (Stretch 2). Fingers point straight out along the arm,
#           so lower to the object's height first, then extend the arm around it.
#   "top":  dex wrist pitched down. Extend over the object first, then lower onto it.
GRASP_MODE = "top" if HAS_DEX_WRIST else "side"

# ------------------------------------------------------------ ROS names
TRAJECTORY_ACTION = "/stretch_controller/follow_joint_trajectory"
JOINT_STATES_TOPIC = "/stretch/joint_states"
MODE_SERVICES = {
    "position": "/switch_to_position_mode",
    "navigation": "/switch_to_navigation_mode",
}
STOW_SERVICE = "/stow_the_robot"

# ---------------------------------------------------------- joint names
# The names the Stretch ROS 2 driver accepts (same in the MuJoCo sim driver).
LIFT = "joint_lift"
ARM = "wrist_extension"            # whole telescoping arm, meters
WRIST_YAW = "joint_wrist_yaw"
WRIST_PITCH = "joint_wrist_pitch"  # dex wrist only
WRIST_ROLL = "joint_wrist_roll"    # dex wrist only
GRIPPER_CMD = "gripper_aperture"   # command name used to open/close (meters on the real robot)
GRIPPER_FINGER = "joint_gripper_finger_left"  # name read back from joint states
HEAD_PAN = "joint_head_pan"
HEAD_TILT = "joint_head_tilt"
BASE_TRANSLATE = "translate_mobile_base"   # position mode only, relative (m)
BASE_ROTATE = "rotate_mobile_base"         # position mode only, relative (rad)

# Arm telescope joints in joint states (summed = extension), if wrist_extension is absent.
ARM_SEGMENTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]

LIFT_LIMITS = (0.0, 1.10)   # m
ARM_LIMITS = (0.0, 0.52)    # m

BASE_FRAME = "base_link"
GRASP_FRAME = "link_grasp_center"

# ---------------------------------------------------------------- poses
SAFE_LIFT = 0.60            # lift height to travel at, arm retracted (m)
if HAS_DEX_WRIST:
    WRIST_GRASP_POSE = {WRIST_YAW: 0.0, WRIST_PITCH: -math.pi / 2, WRIST_ROLL: 0.0}  # CALIBRATE sign
else:
    WRIST_GRASP_POSE = {WRIST_YAW: 0.0}   # gripper straight out along the arm, CALIBRATE

# ------------------------------------------------------------- grasping
# Where on the object the grasp center aims, measured up from the floor point
# perception reports (LitterObject.position is the object's center ON the floor):
# this fraction of the object's height (0.5 = halfway up), but never below GRASP_MIN_HEIGHT.
GRASP_HEIGHT_FRACTION = 0.5
GRASP_MIN_HEIGHT = 0.015    # m above the floor, CALIBRATE
GRASP_Z_OFFSET = 0.0        # extra height on top of the above (m), CALIBRATE
GRASP_WIDTH_MARGIN = 0.02   # open this much wider than the object (m)
REACH_X_TOLERANCE = 0.03    # object may be this far ahead/behind the arm line (m)
MIN_SIDE_DISTANCE = 0.0     # object must be at least this far out from the retracted gripper (m)
GRASP_SETTLE_S = 0.5        # wait after closing/opening before checking (s)
LIFT_CONTACT_EFFORT = 30.0  # contact threshold when lowering, CALIBRATE
DEFAULT_RETRIES = 2         # extra grasp attempts after the first

# Gripper numbers differ between sim and the real robot. Fill them in from calibrate_gripper.
GRIPPER_BY_PLATFORM = {
    "real": {
        "open": 0.09,                # gripper_aperture for fully open (m)
        "closed": -0.03,             # below 0 = keep squeezing, CALIBRATE
        "min_open": 0.03,
        "max_open": 0.09,            # widest graspable object is about this, CALIBRATE
        "empty_closed_finger": 0.0,  # finger reading when closed on nothing, CALIBRATE
        "holding_tolerance": 0.03,   # finger must stop this far above the empty value, CALIBRATE
        "min_holding_effort": 0.0,   # |effort| when squeezing; 0 = effort check off, CALIBRATE
    },
    "sim": {
        "open": 0.09,                # CALIBRATE (the sim passes this straight to MuJoCo)
        "closed": -0.03,             # CALIBRATE
        "min_open": 0.03,
        "max_open": 0.09,
        "empty_closed_finger": 0.0,  # CALIBRATE
        "holding_tolerance": 0.03,   # CALIBRATE
        "min_holding_effort": 0.0,   # sim effort may not be meaningful
    },
}
GRIPPER = GRIPPER_BY_PLATFORM[PLATFORM]

# -------------------------------------------------------------- dropping
BIN_DROP_LIFT = 0.80        # lift height that clears the bin rim (m), CALIBRATE
BIN_DROP_ARM = 0.25         # arm extension that puts the gripper over the opening (m), CALIBRATE
