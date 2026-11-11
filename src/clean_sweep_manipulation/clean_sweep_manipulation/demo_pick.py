"""Pick one object, then (optionally) test the bin drop.

Target from perception (or its ground-truth stub in sim): picks the nearest
CONFIRMED object from litter/detect.

    ros2 run clean_sweep_manipulation demo_pick --ros-args -p source:=perception

Target typed in: object center ON THE FLOOR, in `frame` (default base_link), plus its size.

    ros2 run clean_sweep_manipulation demo_pick --ros-args \
        -p x:=0.0 -p y:=-0.55 -p z:=0.0 -p width:=0.066 -p height:=0.12

align_base defaults to true here (manipulation turns/drives the base itself);
pass -p align_base:=false once navigation does the final approach.
"""

import math

from clean_sweep_manipulation.robot_io import _wait, make_robot


def _param(node, name, default):
    p = node.get_parameter_or(name, None)
    return default if p is None or p.value is None else p.value


def _nearest_confirmed(robot):
    from clean_sweep_interfaces.msg import LitterObject
    from clean_sweep_interfaces.srv import DetectLitter

    client = robot.node.create_client(DetectLitter, "litter/detect")
    if not client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError("litter/detect not available: start perception or ground_truth_stub")
    res = _wait(client.call_async(DetectLitter.Request()), 10.0)
    objs = [o for o in (res.objects if res else []) if o.state == LitterObject.STATE_CONFIRMED]
    if not objs:
        raise RuntimeError("no CONFIRMED litter objects")

    def dist(o):
        p = robot.to_base_frame((o.position.point.x, o.position.point.y, o.position.point.z),
                                o.position.header.frame_id)
        return math.hypot(p[0], p[1])

    return min(objs, key=dist)


def main():
    robot = make_robot("clean_sweep_demo_pick")
    node = robot.node
    align = bool(_param(node, "align_base", True))

    if _param(node, "source", "manual") == "perception":
        litter = _nearest_confirmed(robot)
        print(f"Target: id={litter.id} {litter.label} width={litter.width:.3f} "
              f"height={litter.height:.3f} in {litter.position.header.frame_id}")
        out = robot.pick_litter(litter, align_base=align)
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


if __name__ == "__main__":
    main()
