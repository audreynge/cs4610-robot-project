"""ROS 2 side of manipulation: talks to stretch_driver (robot) or the MuJoCo sim driver.

Same pattern as clean_sweep_perception.head_controller: plain rclpy, the
FollowJointTrajectory action, blocking waits on futures. No hello_helpers, so it
builds in the team's Docker image and runs on Humble or Jazzy.

Two ways to use it:

    # 1. Standalone (demos, scripts): owns its node and spins it in a thread.
    from clean_sweep_manipulation.robot_io import make_robot
    robot = make_robot("clean_sweep_manipulation")      # -> StretchControl
    robot.pick_litter(litter_object)

    # 2. Inside another node (e.g. the state machine): pass your node + a reentrant
    #    callback group, and run that node on a MultiThreadedExecutor. Call robot
    #    methods from callbacks in that group (they block until motions finish).
    io = RosStretchIO(node, callback_group)
    robot = StretchControl(io)
"""

from __future__ import annotations

import atexit
import threading
import time

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from clean_sweep_manipulation import config as C
from clean_sweep_manipulation.stretch_control import StretchControl


class RosStretchIO:
    """Implements the `io` interface StretchControl expects."""

    def __init__(self, node: Node, callback_group: CallbackGroup, motion_timeout: float = 30.0):
        self._node = node
        self._timeout = motion_timeout
        self._joint_state = JointState()
        self._lock = threading.Lock()

        self._traj = ActionClient(node, FollowJointTrajectory, C.TRAJECTORY_ACTION,
                                  callback_group=callback_group)
        node.create_subscription(JointState, C.JOINT_STATES_TOPIC, self._on_joints, 10,
                                 callback_group=callback_group)
        self._modes = {name: node.create_client(Trigger, srv, callback_group=callback_group)
                       for name, srv in C.MODE_SERVICES.items()}
        self._stow = node.create_client(Trigger, C.STOW_SERVICE, callback_group=callback_group)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

    # ------------------------------------------------------------ state
    def _on_joints(self, msg: JointState):
        with self._lock:
            self._joint_state = msg

    def joint_names(self):
        with self._lock:
            return list(self._joint_state.name)

    def joint(self, name):
        with self._lock:
            js = self._joint_state
            if name not in js.name:
                raise KeyError(f"{name} not in {C.JOINT_STATES_TOPIC}: {list(js.name)}")
            i = list(js.name).index(name)
            vel = js.velocity[i] if len(js.velocity) > i else 0.0
            eff = js.effort[i] if len(js.effort) > i else 0.0
            return float(js.position[i]), float(vel), float(eff)

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait for the trajectory action server and the first joint state."""
        if not self._traj.wait_for_server(timeout_sec=timeout):
            self.log("error", f"{C.TRAJECTORY_ACTION} not available; is stretch_driver running?")
            return False
        t0 = time.time()
        while not self.joint_names() and time.time() - t0 < timeout:
            time.sleep(0.1)
        if not self.joint_names():
            self.log("error", f"no messages on {C.JOINT_STATES_TOPIC}")
            return False
        return True

    def lookup(self, target_frame, source_frame):
        """((tx,ty,tz), (qx,qy,qz,qw)): source_frame's pose in target_frame, or None."""
        try:
            t = self._tf_buffer.lookup_transform(target_frame, source_frame, Time(),
                                                 timeout=Duration(seconds=2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.log("warn", f"TF {target_frame} <- {source_frame}: {e}")
            return None
        tr, q = t.transform.translation, t.transform.rotation
        return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------ motion
    def move(self, pose, efforts=None) -> bool:
        """Blocking position-mode move, like HelloNode.move_to_pose.

        efforts: optional {joint: contact threshold}. Stretch reads point.effort as a
        contact threshold; if any joint has one, every joint in the call must.
        """
        if not pose:
            return True
        names = list(pose)
        point = JointTrajectoryPoint(positions=[float(pose[n]) for n in names])
        point.time_from_start = DurationMsg(sec=0, nanosec=0)
        if efforts:
            missing = [n for n in names if n not in efforts]
            if missing:
                raise ValueError(f"contact effort given for some joints but not {missing}")
            point.effort = [float(efforts[n]) for n in names]

        goal = FollowJointTrajectory.Goal()
        goal.goal_time_tolerance = DurationMsg(sec=1, nanosec=0)
        goal.trajectory.joint_names = names
        goal.trajectory.points = [point]

        handle = _wait(self._traj.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self.log("error", f"move {pose} was rejected")
            return False
        result = _wait(handle.get_result_async(), self._timeout)
        if result is None:
            handle.cancel_goal_async()
            self.log("error", f"move {pose} timed out")
            return False
        # error_code defaults to SUCCESSFUL, so aborted/cancelled goals are caught by status.
        if (result.status != GoalStatus.STATUS_SUCCEEDED
                or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
            # A contact-threshold stop can end a lift move early; that's expected when lowering.
            self.log("warn", f"move {pose} ended with status {result.status}, "
                             f"error {result.result.error_code} {result.result.error_string}")
            return False
        return True

    def switch_mode(self, name) -> bool:
        return self._trigger(self._modes[name], C.MODE_SERVICES[name])

    def stow(self) -> bool:
        return self._trigger(self._stow, C.STOW_SERVICE)

    def _trigger(self, client, srv_name) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.log("error", f"{srv_name} not available")
            return False
        res = _wait(client.call_async(Trigger.Request()), 10.0)
        if res is None or not res.success:
            self.log("warn", f"{srv_name} failed: {getattr(res, 'message', 'timeout')}")
            return False
        return True

    # ------------------------------------------------------------ misc
    def sleep(self, seconds):
        time.sleep(seconds)

    def log(self, level, message):
        # One call per severity on its own line: rclpy raises "Logger severity cannot be
        # changed between calls" if a single call site logs at different levels.
        logger = self._node.get_logger()
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warning(message)
        else:
            logger.info(message)


def _wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return future.result() if done.wait(timeout) else None


def make_robot(node_name="clean_sweep_manipulation", args=None):
    """Standalone setup for scripts: init rclpy, spin a node in the background,
    wait for the driver, and return a ready StretchControl (with .node attached)."""
    if not rclpy.ok():
        rclpy.init(args=args)
    node = Node(node_name, automatically_declare_parameters_from_overrides=True)
    group = ReentrantCallbackGroup()
    io = RosStretchIO(node, group)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown():
        # Stop spinning before rclpy shuts down, or the process aborts on exit
        # ("terminate called without an active exception").
        executor.shutdown(timeout_sec=1.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    atexit.register(shutdown)
    if not io.wait_until_ready():
        raise RuntimeError("Stretch driver not reachable (see log above)")
    robot = StretchControl(io)
    robot.node = node
    robot.shutdown = shutdown
    return robot
