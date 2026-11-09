"""Compare the live litter/objects against tape-measured ground truth and print a report.

    ros2 run clean_sweep_perception evaluate_perception --ros-args -p truth_file:=/path/to/truth.yaml

truth.yaml uses the sim stub's format (frame_id + objects with label/position/width/height).
Run it after a `litter/scan` so the objects have had time to be confirmed.
"""

from __future__ import annotations

import time

import rclpy
from clean_sweep_interfaces.msg import LitterObject, LitterObjectArray
from rclpy.node import Node

from clean_sweep_perception.evaluation import Detected, evaluate
from clean_sweep_perception.stub_scene import load_stub_scene

COUNTED_STATES = {LitterObject.STATE_CONFIRMED, LitterObject.STATE_TARGETED, LitterObject.STATE_FAILED}


def main(args=None):
    rclpy.init(args=args)
    node = Node("evaluate_perception")
    truth_file = node.declare_parameter("truth_file", "").value
    radius = node.declare_parameter("match_radius", 0.25).value
    wait = node.declare_parameter("timeout", 10.0).value
    if not truth_file:
        raise SystemExit("parameter 'truth_file' is required")
    scene = load_stub_scene(truth_file)

    latest: list[LitterObjectArray] = []
    node.create_subscription(LitterObjectArray, "litter/objects", latest.append, 10)
    deadline = time.monotonic() + wait
    while not latest and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    if not latest:
        raise SystemExit("no litter/objects received; is litter_registry running?")

    msg = latest[-1]
    if msg.header.frame_id != scene.frame_id:
        node.get_logger().warn(f"truth is in {scene.frame_id!r} but objects are in {msg.header.frame_id!r}")
    detected = [Detected(o.id, o.label, (o.position.point.x, o.position.point.y, o.position.point.z))
                for o in msg.objects if o.state in COUNTED_STATES]
    print(evaluate(list(scene.objects), detected, radius).summary())
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
