"""Moves Stretch's head through the stretch_driver FollowJointTrajectory action."""

from __future__ import annotations

import threading

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

HEAD_JOINTS = ["joint_head_pan", "joint_head_tilt"]


class HeadController:
    """Blocking head moves. Call from a callback in a reentrant group on a
    multi-threaded executor, so the action's own callbacks can still run."""

    def __init__(self, node: Node, action_name: str, callback_group: CallbackGroup,
                 move_duration: float = 1.5):
        self._node = node
        self._client = ActionClient(node, FollowJointTrajectory, action_name, callback_group=callback_group)
        self._move_duration = move_duration

    def move(self, pan: float, tilt: float, timeout: float = 15.0) -> bool:
        if not self._client.wait_for_server(timeout_sec=2.0):
            self._node.get_logger().error("head trajectory action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = HEAD_JOINTS
        point = JointTrajectoryPoint(positions=[float(pan), float(tilt)])
        seconds = int(self._move_duration)
        point.time_from_start = DurationMsg(sec=seconds, nanosec=int((self._move_duration - seconds) * 1e9))
        goal.trajectory.points = [point]

        goal_handle = self._wait(self._client.send_goal_async(goal), timeout)
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error(f"head move to pan={pan:.2f} tilt={tilt:.2f} was rejected")
            return False
        result = self._wait(goal_handle.get_result_async(), timeout)
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self._node.get_logger().error(f"head move to pan={pan:.2f} tilt={tilt:.2f} failed")
            return False
        return True

    @staticmethod
    def _wait(future, timeout: float):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return future.result() if done.wait(timeout) else None
