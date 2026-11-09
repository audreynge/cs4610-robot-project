"""Publishes a photo as if the Stretch head camera were seeing it (for testing without the robot).

Publishes the same topics and TF the real camera stack provides, so the litter
detector runs unchanged:
    /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info
    TF: map -> base_link (identity), base_link -> camera_color_optical_frame

    ros2 run clean_sweep_perception fake_head_camera --ros-args -p image:=/path/to/photo.jpg
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from clean_sweep_perception.fake_camera import fake_frame
from clean_sweep_perception.ros_conversions import matrix_to_quaternion

OPTICAL_FRAME = "camera_color_optical_frame"


class FakeHeadCameraNode(Node):
    def __init__(self):
        super().__init__("fake_head_camera")
        image_path = self.declare_parameter("image", "").value
        rotation = self.declare_parameter("image_rotation", "cw90").value
        height = self.declare_parameter("camera_height", 1.2).value
        pitch = self.declare_parameter("camera_pitch_deg", 45.0).value
        rate = self.declare_parameter("rate", 5.0).value

        upright = cv2.imread(image_path) if image_path else None
        if upright is None:
            raise RuntimeError(f"could not read image {image_path!r} (set parameter 'image')")
        raw, depth, self._camera, pose = fake_frame(upright, rotation, height, pitch)
        self._raw_rgb = raw[:, :, ::-1].copy()
        self._depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)

        self._color_pub = self.create_publisher(Image, "/camera/color/image_raw", 1)
        self._depth_pub = self.create_publisher(Image, "/camera/aligned_depth_to_color/image_raw", 1)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 1)

        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([
            self._transform("map", "base_link", np.eye(3), np.zeros(3)),
            self._transform("base_link", OPTICAL_FRAME, pose.rotation, pose.translation),
        ])
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f"Publishing {image_path} as a {rotation} head camera frame")

    def _transform(self, parent, child, rotation, translation) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = parent, child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, translation)
        q = tf.transform.rotation
        q.x, q.y, q.z, q.w = map(float, matrix_to_quaternion(rotation))
        return tf

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        h, w = self._depth_mm.shape

        color = Image(height=h, width=w, encoding="rgb8", step=w * 3, data=self._raw_rgb.tobytes())
        depth = Image(height=h, width=w, encoding="16UC1", step=w * 2, data=self._depth_mm.tobytes())
        info = CameraInfo(height=h, width=w, distortion_model="plumb_bob")
        c = self._camera
        info.k = [c.fx, 0.0, c.cx, 0.0, c.fy, c.cy, 0.0, 0.0, 1.0]
        info.p = [c.fx, 0.0, c.cx, 0.0, 0.0, c.fy, c.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        for msg in (color, depth, info):
            msg.header.stamp = stamp
            msg.header.frame_id = OPTICAL_FRAME
        self._color_pub.publish(color)
        self._depth_pub.publish(depth)
        self._info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = FakeHeadCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
