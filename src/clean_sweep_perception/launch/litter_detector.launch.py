"""Litter detector on the head camera.

    ros2 launch clean_sweep_perception litter_detector.launch.py \
        detector_model:=/path/to/detector_int8_openvino_model \
        classifier_model:=/path/to/classifier_int8_openvino_model

Before the litter models are trained, a stock COCO model works for bottles and cups:
    detector_model:=yolo26n.pt detector_classes:="['bottle', 'cup']"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = {
    "detector_model": "",
    "classifier_model": "",
    "detector_classes": "['']",
    "detector_confidence": "0.35",
    "image_rotation": "cw90",
    "output_frame": "map",
    "color_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "info_topic": "/camera/color/camera_info",
}


def generate_launch_description():
    def arg(name, value_type=str):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in ARGUMENTS.items()],
        Node(
            package="clean_sweep_perception",
            executable="litter_detector",
            parameters=[{
                "detector_model": arg("detector_model"),
                "classifier_model": arg("classifier_model"),
                "detector_classes": arg("detector_classes", value_type=None),
                "detector_confidence": arg("detector_confidence", value_type=float),
                "image_rotation": arg("image_rotation"),
                "output_frame": arg("output_frame"),
                "color_topic": arg("color_topic"),
                "depth_topic": arg("depth_topic"),
                "info_topic": arg("info_topic"),
            }],
            output="screen",
        ),
    ])
