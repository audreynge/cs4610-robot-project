"""Step through each motion once, pressing Enter between steps.

    ros2 run clean_sweep_manipulation demo_arm_gripper

Run in sim first, then on the robot with a clear area and someone at the run-stop.
"""

from clean_sweep_manipulation import config as C
from clean_sweep_manipulation.robot_io import make_robot


def main():
    robot = make_robot("clean_sweep_demo_arm")
    robot.position_mode()
    print(f"platform={C.PLATFORM} grasp_mode={robot.grasp_mode}")
    print(f"joints: {robot.io.joint_names()}")

    def step(label, fn):
        input(f"\n[Enter] {label} ")
        ok = fn()
        print(f"  ok={ok}  lift={robot.lift_height():.3f} m  arm={robot.arm_extension():.3f} m  "
              f"finger={robot.gripper_finger()[0]:.3f}")

    step("lift to 0.6 m", lambda: robot.set_lift(0.6))
    step("extend arm to 0.2 m", lambda: robot.set_arm(0.2))
    step("wrist to grasp pose (straight out, or down on a dex wrist)", robot.wrist_grasp_pose)
    step("open gripper", robot.open_gripper)
    step("close gripper", robot.close_gripper)
    step("print grasp center in base_link",
         lambda: print("  grasp center:", tuple(round(v, 3) for v in robot.grasp_center_in_base())))
    step("retract arm", robot.retract_arm)
    step("lower lift to 0.2 m, stopping on contact", lambda: robot.lower_lift_until_contact(0.2))
    step("print grasp center in base_link (low)",
         lambda: print("  grasp center:", tuple(round(v, 3) for v in robot.grasp_center_in_base())))
    step("travel pose", robot.travel_pose)
    print("\nDone. Note the grasp-center numbers: navigation needs them for the final approach.")


if __name__ == "__main__":
    main()
