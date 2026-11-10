"""Litter detector on the head camera.

    ros2 launch clean_sweep_perception litter_detector.launch.py

By default it uses the INT8 models bundled in the package (models/), made for the
robot's Intel CPU. On a Mac (Docker is ARM there), pass the .pt files instead, which are faster:
    detector_model:=<share>/models/detector.pt classifier_model:=<share>/models/classifier.pt
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from clean_sweep_perception.litter_detector_node import PARAMETER_DEFAULTS

MODELS = os.path.join(get_package_share_directory("clean_sweep_perception"), "models")
# Every node parameter is a launch argument, with the node's own default; only the models differ.
DEFAULTS = {
    **PARAMETER_DEFAULTS,
    "detector_model": os.path.join(MODELS, "detector_int8_openvino_model"),
    "classifier_model": os.path.join(MODELS, "classifier_int8_openvino_model"),
}


def generate_launch_description():
    def parameter(name):
        default = DEFAULTS[name]
        # Lists are parsed from their YAML text (value_type=None); everything else keeps its type.
        value_type = None if isinstance(default, list) else type(default)
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=str(value)) for name, value in DEFAULTS.items()],
        Node(
            package="clean_sweep_perception",
            executable="litter_detector",
            parameters=[{name: parameter(name) for name in DEFAULTS}],
            output="screen",
        ),
    ])
