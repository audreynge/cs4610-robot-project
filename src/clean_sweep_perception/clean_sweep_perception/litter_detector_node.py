"""Detects litter in the head camera stream and publishes per-frame 3D detections.

Subscribes (time-synchronized):
    color_topic  sensor_msgs/Image        raw head camera color image
    depth_topic  sensor_msgs/Image        depth aligned to the color image
    info_topic   sensor_msgs/CameraInfo   color intrinsics
Publishes:
    litter/detections   clean_sweep_interfaces/LitterObjectArray  this frame's accepted detections
                        (state SEEN, id 0), positions in `output_frame`
    litter/debug_image  sensor_msgs/Image  upright image with accepted/rejected boxes

Tracking across frames (ids, confirmation) is the registry node's job.
"""

from __future__ import annotations

import time

import message_filters
import rclpy
import tf2_ros
from clean_sweep_interfaces.msg import LitterObject, LitterObjectArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from clean_sweep_perception.debug_view import draw_debug
from clean_sweep_perception.geometry import CameraModel
from clean_sweep_perception.models import YoloClassifier, YoloDetector
from clean_sweep_perception.pipeline import LitterPerception
from clean_sweep_perception.ros_conversions import image_to_numpy, transform_from_tf


class LitterDetectorNode(Node):
    def __init__(self):
        super().__init__("litter_detector")
        p = self._declare_parameters()

        if not p["detector_model"]:
            raise RuntimeError("parameter 'detector_model' is required")
        detector = YoloDetector(p["detector_model"], p["detector_confidence"], p["detector_imgsz"],
                                [c for c in p["detector_classes"] if c] or None)
        classifier = YoloClassifier(p["classifier_model"]) if p["classifier_model"] else None
        self._perception = LitterPerception(detector, classifier, p["image_rotation"])
        self._base_frame = p["base_frame"]
        self._output_frame = p["output_frame"]
        self._min_period = 1.0 / p["max_rate"]
        self._last_run = 0.0

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._detections_pub = self.create_publisher(LitterObjectArray, "litter/detections", 10)
        self._debug_pub = self.create_publisher(Image, "litter/debug_image", 1)

        subscribers = [
            message_filters.Subscriber(self, Image, p["color_topic"]),
            message_filters.Subscriber(self, Image, p["depth_topic"]),
            message_filters.Subscriber(self, CameraInfo, p["info_topic"]),
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=5, slop=0.1)
        self._sync.registerCallback(self._on_frame)
        self.get_logger().info(
            f"Detector {p['detector_model']}, classifier {p['classifier_model'] or 'none'}, "
            f"listening on {p['color_topic']}"
        )

    def _declare_parameters(self) -> dict:
        defaults = {
            "color_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "info_topic": "/camera/color/camera_info",
            "base_frame": "base_link",
            "output_frame": "map",
            "detector_model": "",
            "detector_confidence": 0.35,
            "detector_imgsz": 640,
            "detector_classes": [""],  # e.g. ["bottle", "cup"] for a stock COCO model
            "classifier_model": "",
            "image_rotation": "cw90",  # how to turn the raw head image upright
            "max_rate": 3.0,  # Hz; the litter doesn't move, so spare the CPU
        }
        return {name: self.declare_parameter(name, value).value for name, value in defaults.items()}

    def _lookup(self, target: str, source: str, stamp):
        tf = self._tf_buffer.lookup_transform(target, source, stamp, timeout=Duration(seconds=0.2))
        return transform_from_tf(tf.transform.translation, tf.transform.rotation)

    def _on_frame(self, color: Image, depth: Image, info: CameraInfo) -> None:
        now = time.monotonic()
        if now - self._last_run < self._min_period:
            return
        self._last_run = now

        stamp = Time.from_msg(color.header.stamp)
        try:
            camera_to_base = self._lookup(self._base_frame, color.header.frame_id, stamp)
            base_to_output = self._lookup(self._output_frame, self._base_frame, stamp)
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF not available: {e}", throttle_duration_sec=5.0)
            return

        color_bgr = image_to_numpy(color.encoding, color.height, color.width, color.step, color.data)
        depth_m = image_to_numpy(depth.encoding, depth.height, depth.width, depth.step, depth.data)
        camera = CameraModel(fx=info.k[0], fy=info.k[4], cx=info.k[2], cy=info.k[5])
        result = self._perception.process(color_bgr, depth_m, camera, camera_to_base)

        out = LitterObjectArray()
        out.header.stamp = color.header.stamp
        out.header.frame_id = self._output_frame
        for detection, loc in result.accepted:
            obj = LitterObject()
            obj.label = detection.label or "other"
            obj.label_confidence = detection.label_confidence
            obj.detection_confidence = detection.confidence
            obj.position.header = out.header
            x, y, z = base_to_output.points(loc.position[None, :])[0]
            obj.position.point.x, obj.position.point.y, obj.position.point.z = float(x), float(y), float(z)
            obj.width, obj.height = float(loc.width), float(loc.height)
            obj.depth_valid = loc.depth_valid
            obj.state = LitterObject.STATE_SEEN
            out.objects.append(obj)
        self._detections_pub.publish(out)

        if self._debug_pub.get_subscription_count():
            self._debug_pub.publish(self._to_image_msg(draw_debug(result), color.header))

    @staticmethod
    def _to_image_msg(bgr, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = bgr.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LitterDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
