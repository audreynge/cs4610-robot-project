# clean_sweep_manipulation

Lift, arm, wrist and gripper control for Stretch, plus the two things the state
machine calls: **pick a litter object** and **drop it in a bin**. Works with the
Stretch 2 standard gripper (default) and a dex wrist.

```
litter/refine ──LitterObject──> pick_litter() ──result──> state machine ──> litter/verify_pickup
                                     │
                                     └─ FollowJointTrajectory, joint states, TF ──> stretch_driver
```

## Interface for the rest of the team

```python
from clean_sweep_manipulation.robot_io import RosStretchIO
from clean_sweep_manipulation.stretch_control import StretchControl

io = RosStretchIO(node, reentrant_callback_group)   # inside your node (MultiThreadedExecutor)
robot = StretchControl(io)
# or, in a standalone script:  from clean_sweep_manipulation.robot_io import make_robot; robot = make_robot()
```

| Call | What it does |
|---|---|
| `pick_litter(litter)` | Picks a `LitterObject` (use the one `litter/refine` returns). Uses its `position` (any frame with TF to `base_link`, normally `map`), `width` and `height`. Returns a result dict (below). |
| `drop_in_bin()` | Bin on the arm side: raise above the rim, reach over, open, retract, then close to check the gripper is empty. Returns `True` if released. |
| `position_mode()` / `navigation_mode()` | Switch the driver. Pick and drop switch to position mode themselves; switch back before Nav2 drives. |
| `travel_pose()` | Arm in, lift at a safe height: before driving. |
| `is_holding()` | Grasp check: finger position **and** effort. |
| `check_object_reachable(point)` | Is an object (base_link) beside the arm and in reach? |

**Result codes** (`result["result"]`):

| Code | Meaning | What the state machine does |
|---|---|---|
| `SUCCESS` | Object held, arm retracted (out of the camera's view) | `litter/verify_pickup`, then deliver |
| `GRASP_FAILED` | Failed 1 + 2 retries (opens wider each time) | `litter/set_state FAILED`, next object |
| `NEEDS_REPOSITION` | Not beside the arm or out of reach; `result["reach"]["dx"]` = drive forward this much (m) | Navigation adjusts, then `refine` + pick again |
| `TOO_LARGE` | Wider than the gripper opens (about 9 cm). No motion. | `set_state FAILED`, next object |
| `ERROR` | e.g. no TF from the object's frame (is `map` running?); see `result["message"]` | Retry or skip |

## What manipulation needs from navigation

Park so the object is **on the arm side** (the robot's right, `-y` in `base_link`),
**lined up with the arm** (within `REACH_X_TOLERANCE`, 3 cm, along x), and **no further
out than full arm reach**. Get the exact numbers from `demo_arm_gripper`: it prints
the grasp center with the arm retracted; add up to 0.52 m of reach. For the bin,
park with the bin on the arm side too.

Until navigation's final approach exists, picks can align the base themselves:
`pick_litter(litter, align_base=True)`.

## How a pick works

1. Too wide for the gripper? Return `TOO_LARGE` without moving.
2. Position mode; travel pose (arm in, lift up, gripper straight out, or down on a dex wrist).
3. Aim point: the object's floor center, raised to half its height (`GRASP_HEIGHT_FRACTION`).
4. Not beside the arm or out of reach? Return `NEEDS_REPOSITION`.
5. Open to width + 2 cm.
6. **Side grasp (standard gripper):** lower beside the object (stops on contact), then extend around it.
   **Top grasp (dex wrist):** extend over it, then lower onto it.
7. Close, check the grasp, lift, check again (catches slips). Held: retract, `SUCCESS`.
8. Missed: open, retract, open 1 cm wider, retry.

## Running

Set these first (both have defaults):

```bash
export CLEAN_SWEEP_PLATFORM=real   # or sim: gripper values differ between them
export CLEAN_SWEEP_DEX_WRIST=0     # 1 for a dex wrist (Stretch 3 / upgraded Stretch 2)
```

With `stretch_driver` running in **position mode** (`mode:=position`):

```bash
ros2 run clean_sweep_manipulation demo_arm_gripper     # each joint, step by step
ros2 run clean_sweep_manipulation calibrate_gripper    # measures the gripper numbers
ros2 run clean_sweep_manipulation demo_pick --ros-args -p source:=perception   # nearest CONFIRMED litter
ros2 run clean_sweep_manipulation demo_pick --ros-args -p x:=0.0 -p y:=-0.55 -p width:=0.066 -p height:=0.12
```

`demo_pick` works with the perception stack or its `ground_truth_stub` (in sim).

## Calibrate before trusting a grasp

Every value marked `CALIBRATE` in [`config.py`](clean_sweep_manipulation/config.py) is a guess:

1. **Wrist:** in `demo_arm_gripper`, check the gripper points straight out along the arm (dex wrist: down). Fix `WRIST_GRASP_POSE` if not.
2. **Gripper:** run `calibrate_gripper` with a can, a cup and a paper ball; paste the printed values into `GRIPPER_BY_PLATFORM`.
3. **Floor:** run `demo_pick` on a floor object; tune `GRASP_MIN_HEIGHT`, `GRASP_Z_OFFSET` and `LIFT_CONTACT_EFFORT`.
4. **Bin:** set `BIN_DROP_LIFT` (clears the rim) and `BIN_DROP_ARM` (over the opening).

## Tests

```bash
./run.sh --test   # from the repo root; includes these (no ROS needed)
```

- `test_geometry.py`: reach planning, frames, grasp check math.
- `test_pick_logic.py`: full pick and drop against a fake robot whose gripper only holds the object if the
  grasp center really reaches it. Covers both grasp modes, retries, slips, `TOO_LARGE`,
  `NEEDS_REPOSITION`, picking from the `map` frame and a missing TF.

**Checked so far:** built with colcon on ROS 2 Humble and run end to end against a fake
`stretch_driver` (same action, joint-states topic, TF frames and services): picks from
`base_link` and from `map`, the drop check and a failed-grasp retry all behaved correctly.
**Not yet:** the real robot, the MuJoCo sim, Jazzy, and `demo_pick -p source:=perception`.

## Layout

| File | Contents |
|---|---|
| `stretch_control.py` | All pick/drop logic. No ROS: talks to the robot through an `io` object |
| `robot_io.py` | The ROS 2 side (`RosStretchIO`, `make_robot`), same pattern as perception's `head_controller` |
| `geometry.py` | Pure math (reach plan, frames, grasp check) |
| `config.py` | Every tunable value |
| `demo_*.py`, `calibrate_gripper.py` | Scripts above |
