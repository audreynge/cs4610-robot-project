"""Publishes known litter positions in the same format as the real perception node.

Lets navigation, manipulation and the state machine be developed in simulation
before the detector is ready. Swap this node for the real one without changing
any consumer.

Topics / services:
    litter/objects  (clean_sweep_interfaces/LitterObjectArray), published periodically
    litter/detect   (clean_sweep_interfaces/DetectLitter)
"""

import rclpy
from clean_sweep_interfaces.msg import LitterObject, LitterObjectArray
from clean_sweep_interfaces.srv import DetectLitter
from rclpy.node import Node

from clean_sweep_perception.stub_scene import StubScene, load_stub_scene


class GroundTruthStubNode(Node):
    def __init__(self):
        super().__init__("ground_truth_stub")
        objects_file = self.declare_parameter("objects_file", "").value
        publish_rate = self.declare_parameter("publish_rate", 2.0).value

        if not objects_file:
            raise RuntimeError("parameter 'objects_file' is required")
        self._scene: StubScene = load_stub_scene(objects_file)
        self.get_logger().info(
            f"Loaded {len(self._scene.objects)} objects from {objects_file}"
        )

        self._publisher = self.create_publisher(LitterObjectArray, "litter/objects", 10)
        self.create_service(DetectLitter, "litter/detect", self._on_detect)
        self.create_timer(1.0 / publish_rate, self._publish)

    def _build_objects(self) -> list[LitterObject]:
        stamp = self.get_clock().now().to_msg()
        msgs = []
        for index, obj in enumerate(self._scene.objects):
            msg = LitterObject()
            msg.id = index + 1
            msg.label = obj.label
            msg.label_confidence = 1.0
            msg.detection_confidence = 1.0
            msg.position.header.frame_id = self._scene.frame_id
            msg.position.header.stamp = stamp
            msg.position.point.x, msg.position.point.y, msg.position.point.z = obj.position
            msg.width = obj.width
            msg.height = obj.height
            msg.depth_valid = True
            msg.state = LitterObject.STATE_CONFIRMED
            msgs.append(msg)
        return msgs

    def _publish(self):
        array = LitterObjectArray()
        array.header.frame_id = self._scene.frame_id
        array.header.stamp = self.get_clock().now().to_msg()
        array.objects = self._build_objects()
        self._publisher.publish(array)

    def _on_detect(self, _request, response):
        response.objects = self._build_objects()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
