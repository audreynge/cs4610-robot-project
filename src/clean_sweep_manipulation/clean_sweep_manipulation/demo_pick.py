"""Pick one object, then (optionally) test the bin drop.

Target from perception (or its ground-truth stub in sim): points the head at the floor
on the arm side (-p head_pan:=, -p head_tilt:= in rad), waits up to detect_timeout s for
a CONFIRMED object from litter/detect, and picks the nearest. With align_base it first
drives up to it, re-measuring with litter/refine after every base move (wheels slip), then
grasps with the arm alone. It does that again before each retry (-p attempts:=, default 3),
since a miss can push the object.

    ros2 run clean_sweep_manipulation demo_pick --ros-args -p source:=perception

Target typed in: object center ON THE FLOOR, in `frame` (default base_link), plus its size.

    ros2 run clean_sweep_manipulation demo_pick --ros-args \
        -p x:=0.0 -p y:=-0.55 -p z:=0.0 -p width:=0.066 -p height:=0.12

align_base defaults to true here (manipulation turns/drives the base itself);
pass -p align_base:=false once navigation does the final approach.
At the end it undoes its own base moves (alignment turn, bin-drop back-up) so it ends
where it started; pass -p return_to_start:=false to stay put.
"""

import math
import time

from clean_sweep_manipulation import config as C
from clean_sweep_manipulation.robot_io import _wait, make_robot


def _param(node, name, default):
    p = node.get_parameter_or(name, None)
    return default if p is None or p.value is None else p.value


def _nearest_confirmed(robot, timeout_s=60.0):
    """Poll litter/detect until something is CONFIRMED (the registry needs a few frames)."""
    from clean_sweep_interfaces.msg import LitterObject
    from clean_sweep_interfaces.srv import DetectLitter

    client = robot.node.create_client(DetectLitter, "litter/detect")
    if not client.wait_for_service(timeout_sec=30.0):
        raise RuntimeError("litter/detect not available: start perception or ground_truth_stub")
    deadline = time.monotonic() + timeout_s
    seen = None
    while True:
        res = _wait(client.call_async(DetectLitter.Request()), 10.0)
        found = list(res.objects) if res else []
        objs = [o for o in found if o.state == LitterObject.STATE_CONFIRMED]
        if objs:
            break
        if len(found) != seen:
            seen = len(found)
            print(f"waiting for a CONFIRMED object ({seen} tracked, none confirmed yet)...")
        if time.monotonic() > deadline:
            raise RuntimeError(f"no CONFIRMED litter after {timeout_s:.0f} s "
                               f"({len(found)} tracked). Is it in the camera's view?")
        time.sleep(1.0)

    def dist(o):
        p = robot.to_base_frame((o.position.point.x, o.position.point.y, o.position.point.z),
                                o.position.header.frame_id)
        return math.hypot(p[0], p[1])

    return min(objs, key=dist)


def _base_point(robot, litter):
    p = litter.position.point
    return robot.to_base_frame((p.x, p.y, p.z), litter.position.header.frame_id)


def _approach_and_refine(robot, litter):
    """Drive up beside the object, re-measuring it with litter/refine after every base move.

    The far-away estimate only sees the object's front face, and the wheels slip, so the
    grasp uses a measurement taken after the last base move.
    Returns (LitterObject to pick, True if the base needs no more moves).
    """
    from clean_sweep_interfaces.srv import RefineLitter

    client = robot.node.create_client(RefineLitter, "litter/refine")
    if not client.wait_for_service(timeout_sec=5.0):
        print("litter/refine not available (sim stub?): using the far-away estimate")
        return litter, False
    latest = [litter]

    def measure():
        res = _wait(client.call_async(RefineLitter.Request(id=litter.id)), 30.0)
        if res is None or not res.success:
            print(f"refine failed ({res.message if res else 'timed out'})")
            return None
        before = _base_point(robot, latest[0])
        latest[0] = res.object
        now = _base_point(robot, res.object)
        print(f"refined: {res.message}; moved {math.dist(before[:2], now[:2]) * 100:.1f} cm "
              f"from the previous estimate")
        return now

    robot.position_mode()
    _, settled = robot.approach(_base_point(robot, litter), measure)
    return latest[0], settled


def main():
    robot = make_robot("clean_sweep_demo_pick")
    node = robot.node
    align = bool(_param(node, "align_base", True))

    if _param(node, "source", "manual") == "perception":
        # Look down at the floor on the arm side (pan negative = right, tilt negative = down).
        robot.travel_pose()
        robot.set_head(pan=float(_param(node, "head_pan", -math.pi / 2)),
                       tilt=float(_param(node, "head_tilt", -1.0)))
        litter = _nearest_confirmed(robot, float(_param(node, "detect_timeout", 60.0)))
        far = _base_point(robot, litter)
        print(f"Target: id={litter.id} {litter.label} width={litter.width:.3f} "
              f"height={litter.height:.3f}, at ({far[0]:.3f}, {far[1]:.3f}) in base_link")
        # A missed grasp often pushes the object (paper especially), so each try re-measures
        # it and lines up again instead of grabbing at the old spot.
        tries = int(_param(node, "attempts", C.DEFAULT_RETRIES + 1))
        for attempt in range(1, tries + 1):
            settled = False
            if align:
                litter, settled = _approach_and_refine(robot, litter)
            # Settled: grasp from here with the arm only (a blind base move would undo the
            # measurement). Otherwise let the pick line up the base itself.
            out = robot.pick_litter(litter, retries=0, align_base=align and not settled)
            out["attempts"] = attempt
            if out["result"] != robot.GRASP_FAILED or attempt == tries:
                break
            print(f"attempt {attempt}/{tries} missed: re-measuring the object (it may have moved)")
    else:
        frame = _param(node, "frame", "base_link")
        point = (float(_param(node, "x", 0.0)), float(_param(node, "y", -0.55)),
                 float(_param(node, "z", 0.0)))
        floor = robot.to_base_frame(point, frame)
        out = robot.pick_object_at(floor, object_width=float(_param(node, "width", 0.066)),
                                   object_height=float(_param(node, "height", 0.12)),
                                   align_base=align)

    print(f"RESULT: {out['result']} after {out['attempts']} attempt(s). {out['message']}")
    if out["result"] == robot.NEEDS_REPOSITION and out["reach"]:
        print(f"  drive the base by dx={out['reach']['dx']:.3f} m")
    if out["result"] == robot.SUCCESS and input("Test the bin drop? [y/N] ").lower() == "y":
        print("released:", robot.drop_in_bin())
    if _param(node, "return_to_start", True):
        print("back at the start:", robot.return_to_start())


if __name__ == "__main__":
    main()
