"""Tracks litter across frames and offers the services the state machine uses.

Subscribes:
    litter/detections  (LitterObjectArray)  per-frame detections from litter_detector
Publishes:
    litter/objects     (LitterObjectArray)  every tracked object with id and state
    litter/markers     (visualization_msgs/MarkerArray)  for RViz
Services:
    litter/detect         (DetectLitter)    tracked objects
    litter/set_state      (SetLitterState)  e.g. TARGETED / FAILED
    litter/scan           (ScanForLitter)   sweep the head, return confirmed objects
    litter/refine         (RefineLitter)    look at one object up close, re-estimate it
    litter/verify_pickup  (VerifyPickup)    look where it was; mark PICKED if gone
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from clean_sweep_interfaces.msg import LitterObject, LitterObjectArray
from clean_sweep_interfaces.srv import (
    DetectLitter, RefineLitter, ScanForLitter, SetLitterState, VerifyPickup,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray

from clean_sweep_perception.head import look_at_angles
from clean_sweep_perception.head_controller import HeadController
from clean_sweep_perception.registry import (
    LitterRegistry, Observation, RegistryParams, State, TrackedObject,
)
from clean_sweep_perception.ros_conversions import transform_from_tf
from clean_sweep_perception.sorting import BinRules

@dataclass(frozen=True)
class Look:
    """Result of pointing the head at an object."""

    position: np.ndarray  # where the object was believed to be
    started: float  # monotonic time the head started moving
    observations: list[Observation]  # every detection received after the head settled


STATE_COLORS = {  # RGBA per state for RViz
    State.SEEN: (0.6, 0.6, 0.6, 0.6), State.CONFIRMED: (0.1, 0.8, 0.1, 0.9),
    State.TARGETED: (1.0, 0.6, 0.0, 0.9), State.PICKED: (0.2, 0.4, 1.0, 0.4),
    State.FAILED: (0.9, 0.1, 0.1, 0.9),
}


class LitterRegistryNode(Node):
    def __init__(self):
        super().__init__("litter_registry")
        p = {name: self.declare_parameter(name, value).value for name, value in {
            "frame": "map",  # must match litter_detector's output_frame
            "base_frame": "base_link",
            "head_frame": "link_head_pan",  # its origin is used as the head's pivot
            "pan_zero_heading": 0.0,  # base-frame heading the camera faces at pan = 0 (rad)
            "head_action": "/stretch_controller/follow_joint_trajectory",
            "scan_pans": [-2.4, -1.6, -0.8, 0.0, 0.8],  # rad
            "scan_tilt": -0.75,  # rad, negative looks down
            "dwell": 1.5,  # s to let the detector look after each head move
            "match_radius": RegistryParams.match_radius,
            "confirm_sightings": RegistryParams.confirm_sightings,
            "forget_after": RegistryParams.forget_after,
            "refine_radius": 0.25,  # m; close-range detections this near the object count
            "verify_radius": 0.12,  # m; any detection this near means it's still there
            "min_frames": 2,  # detector frames needed during a look before trusting "not seen"
            "publish_rate": 2.0,
            "bins_file": "",  # default: this package's config/bins.yaml
        }.items()}
        self._p = p
        self._registry = LitterRegistry(RegistryParams(
            match_radius=p["match_radius"], confirm_sightings=p["confirm_sightings"],
            forget_after=p["forget_after"],
        ))
        self._bins = BinRules.from_yaml(p["bins_file"] or os.path.join(
            get_package_share_directory("clean_sweep_perception"), "config", "bins.yaml"))
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, Observation]] = deque(maxlen=500)  # (receive time, obs)
        self._frame_times: deque[float] = deque(maxlen=200)  # receive time of every detections message
        self._head_lock = threading.Lock()  # one head-moving request at a time

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        detections_group = MutuallyExclusiveCallbackGroup()
        services_group = ReentrantCallbackGroup()
        self._head = HeadController(self, p["head_action"], services_group)

        self.create_subscription(LitterObjectArray, "litter/detections", self._on_detections, 10,
                                 callback_group=detections_group)
        self._objects_pub = self.create_publisher(LitterObjectArray, "litter/objects", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "litter/markers", 1)
        self.create_timer(1.0 / p["publish_rate"], self._publish, callback_group=detections_group)

        self.create_service(DetectLitter, "litter/detect", self._on_detect, callback_group=services_group)
        self.create_service(SetLitterState, "litter/set_state", self._on_set_state,
                            callback_group=services_group)
        self.create_service(ScanForLitter, "litter/scan", self._on_scan, callback_group=services_group)
        self.create_service(RefineLitter, "litter/refine", self._on_refine, callback_group=services_group)
        self.create_service(VerifyPickup, "litter/verify_pickup", self._on_verify_pickup,
                            callback_group=services_group)

    # --- tracking -------------------------------------------------------------------------

    def _on_detections(self, msg: LitterObjectArray) -> None:
        if msg.header.frame_id != self._p["frame"]:
            self.get_logger().warn(f"ignoring detections in frame {msg.header.frame_id!r}, "
                                   f"expected {self._p['frame']!r}", throttle_duration_sec=10.0)
            return
        now = time.monotonic()
        observations = [Observation(
            position=np.array([d.position.point.x, d.position.point.y, d.position.point.z]),
            width=d.width, height=d.height, depth_valid=d.depth_valid,
            detection_confidence=d.detection_confidence,
            label=d.label if d.label_confidence > 0 else None, label_confidence=d.label_confidence,
        ) for d in msg.objects]
        with self._lock:
            self._registry.update(observations, now)
            self._recent.extend((now, o) for o in observations)
            self._frame_times.append(now)

    def _observations_since(self, since: float) -> list[Observation]:
        with self._lock:
            return [o for t, o in self._recent if t >= since]

    def _frames_since(self, since: float) -> int:
        """Detector messages (including empty ones) received since `since`."""
        with self._lock:
            return sum(t >= since for t in self._frame_times)

    # --- outputs --------------------------------------------------------------------------

    def _to_msg(self, obj: TrackedObject, stamp) -> LitterObject:
        msg = LitterObject()
        msg.id = obj.id
        msg.label = obj.label
        msg.label_confidence = float(obj.label_confidence)
        msg.detection_confidence = float(obj.detection_confidence)
        msg.bin = self._bins.bin_for(obj.label, obj.label_confidence)
        msg.position.header.frame_id = self._p["frame"]
        msg.position.header.stamp = stamp
        msg.position.point.x, msg.position.point.y, msg.position.point.z = map(float, obj.position)
        msg.width, msg.height = float(obj.width), float(obj.height)
        msg.depth_valid = obj.depth_valid
        msg.state = int(obj.state)
        return msg

    def _object_msgs(self, states=None) -> list[LitterObject]:
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            return [self._to_msg(o, stamp) for o in self._registry.objects()
                    if states is None or o.state in states]

    def _publish(self) -> None:
        array = LitterObjectArray()
        array.header.frame_id = self._p["frame"]
        array.header.stamp = self.get_clock().now().to_msg()
        array.objects = self._object_msgs()
        self._objects_pub.publish(array)
        if self._markers_pub.get_subscription_count():
            self._markers_pub.publish(self._markers(array))

    def _markers(self, array: LitterObjectArray) -> MarkerArray:
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for obj in array.objects:
            body = Marker(type=Marker.CYLINDER, ns="litter", id=obj.id, header=array.header)
            body.pose.position = obj.position.point
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = max(obj.width, 0.02)
            body.scale.z = max(obj.height, 0.02)
            body.color.r, body.color.g, body.color.b, body.color.a = STATE_COLORS[State(obj.state)]
            text = Marker(type=Marker.TEXT_VIEW_FACING, ns="litter_label", id=obj.id, header=array.header)
            text.pose.position.x, text.pose.position.y = obj.position.point.x, obj.position.point.y
            text.pose.position.z = obj.height + 0.08
            text.scale.z = 0.06
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"{obj.id}: {obj.label} [{State(obj.state).name}]"
            markers.markers += [body, text]
        return markers

    # --- services -------------------------------------------------------------------------

    def _on_detect(self, _request, response):
        response.objects = self._object_msgs()
        return response

    def _on_set_state(self, request, response):
        try:
            state = State(request.state)
        except ValueError:
            response.success, response.message = False, f"unknown state {request.state}"
            return response
        with self._lock:
            response.success = self._registry.set_state(request.id, state)
        response.message = "" if response.success else f"no object with id {request.id}"
        return response

    def _on_scan(self, _request, response):
        if not self._head_lock.acquire(blocking=False):
            response.success, response.message = False, "head is busy with another request"
            return response
        try:
            started = time.monotonic()
            for pan in self._p["scan_pans"]:
                if not self._head.move(pan, self._p["scan_tilt"]):
                    response.success, response.message = False, f"head move to pan {pan:.2f} failed"
                    break
                time.sleep(self._p["dwell"])
            else:
                response.success = self._frames_since(started) >= self._p["min_frames"]
                if not response.success:
                    response.message = "no detector frames arrived during the scan; is litter_detector running?"
        finally:
            self._head_lock.release()
        response.objects = self._object_msgs({State.CONFIRMED})
        return response

    def _look_at_object(self, object_id: int) -> tuple[Look | None, str]:
        """Point the head at a tracked object and wait `dwell` (None plus an error on failure)."""
        with self._lock:
            obj = self._registry.get(object_id)
            position = None if obj is None else obj.position.copy()
        if position is None:
            return None, f"no object with id {object_id}"
        try:
            to_base = self._lookup(self._p["base_frame"], self._p["frame"])
            head = self._lookup(self._p["base_frame"], self._p["head_frame"])
        except tf2_ros.TransformException as e:
            return None, f"TF not available: {e}"
        target = to_base.points(position[None, :])[0]
        pan, tilt = look_at_angles(target, head.translation, self._p["pan_zero_heading"])

        if not self._head_lock.acquire(blocking=False):
            return None, "head is busy with another request"
        try:
            started = time.monotonic()
            if not self._head.move(pan, tilt):
                return None, "head move failed"
            settled = time.monotonic()
            time.sleep(self._p["dwell"])
        finally:
            self._head_lock.release()

        frames = self._frames_since(settled)
        if frames < self._p["min_frames"]:
            return None, (f"only {frames} detector frames arrived while looking "
                          f"(need {self._p['min_frames']}); is litter_detector running?")
        return Look(position, started, self._observations_since(settled)), ""

    def _on_refine(self, request, response):
        look, error = self._look_at_object(request.id)
        if look is None:
            response.success, response.message = False, error
            return response
        with self._lock:
            used = self._registry.refine(request.id, look.observations, self._p["refine_radius"],
                                         since=look.started, now=time.monotonic())
            obj = self._registry.get(request.id)
            if obj is None:
                response.success, response.message = False, f"object {request.id} was forgotten"
                return response
            response.object = self._to_msg(obj, self.get_clock().now().to_msg())
        response.success = used > 0
        response.message = (f"re-estimated from {used} close-range detections" if used
                            else "not seen up close; it may have been a false positive")
        return response

    def _on_verify_pickup(self, request, response):
        look, error = self._look_at_object(request.id)
        if look is None:
            response.picked_up, response.message = False, error
            return response
        still_there = [o for o in look.observations
                       if np.linalg.norm(o.position[:2] - look.position[:2]) <= self._p["verify_radius"]]
        if still_there:
            response.picked_up, response.message = False, f"still seen in {len(still_there)} detections"
            return response
        with self._lock:
            marked = self._registry.set_state(request.id, State.PICKED)
        response.picked_up = marked
        response.message = ("gone from the floor; marked PICKED" if marked
                            else f"object {request.id} was forgotten")
        return response

    def _lookup(self, target: str, source: str):
        tf = self._tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=1.0))
        return transform_from_tf(tf.transform.translation, tf.transform.rotation)


def main(args=None):
    rclpy.init(args=args)
    node = LitterRegistryNode()
    executor = MultiThreadedExecutor(num_threads=8)  # services block while the head moves
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
