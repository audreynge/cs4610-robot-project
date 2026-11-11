"""Measure the gripper numbers for config.GRIPPER_BY_PLATFORM.

    ros2 run clean_sweep_manipulation calibrate_gripper

Closes on nothing, then on an object you hold between the fingers, and prints
empty_closed_finger, holding_tolerance and min_holding_effort to paste into config.py.
Repeat with a can, a cup and a paper ball; keep the SMALLEST difference.
"""

from clean_sweep_manipulation import config as C
from clean_sweep_manipulation.robot_io import make_robot


def main():
    robot = make_robot("clean_sweep_calibrate")
    robot.position_mode()
    print(f"platform={C.PLATFORM}  open={C.GRIPPER['open']}  closed={C.GRIPPER['closed']}")

    def reading(label):
        robot.io.sleep(0.8)
        pos, eff = robot.gripper_finger()
        print(f"  {label}: position={pos:.4f}  effort={eff:.2f}")
        return pos, eff

    robot.open_gripper()
    reading("open")
    input("\nGripper EMPTY? Press Enter to close...")
    robot.close_gripper()
    empty, empty_eff = reading("closed on nothing")

    robot.open_gripper()
    input("\nPut an object between the fingers, then press Enter to close...")
    robot.close_gripper()
    full, full_eff = reading("closed on object")
    robot.open_gripper()

    diff = full - empty
    print(f"\nPaste into config.GRIPPER_BY_PLATFORM['{C.PLATFORM}']:")
    print(f"    'empty_closed_finger': {empty:.4f},")
    print(f"    'holding_tolerance': {max(diff / 2, 0.005):.4f},   # half of {diff:.4f}")
    if abs(full_eff) > abs(empty_eff) + 2.0:
        print(f"    'min_holding_effort': {(abs(full_eff) + abs(empty_eff)) / 2:.1f},")
    else:
        print("    'min_holding_effort': 0.0,   # effort didn't change clearly; leave the check off")
    if diff <= 0.01:
        print("WARNING: empty and holding readings are almost equal. Check the open/closed "
              "values, or try a bigger object.")


if __name__ == "__main__":
    main()
