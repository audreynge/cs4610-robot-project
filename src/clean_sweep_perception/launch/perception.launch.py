"""Full perception stack: litter detector + registry (tracking and services).

    ros2 launch clean_sweep_perception perception.launch.py

The bundled models are used unless detector_model:=... / classifier_model:=... are given.
Other arguments pass through to litter_detector.launch.py (e.g. detector_classes,
image_rotation, output_frame).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("clean_sweep_perception")
    frame = LaunchConfiguration("output_frame")
    return LaunchDescription([
        DeclareLaunchArgument("output_frame", default_value="map"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "litter_detector.launch.py")),
            launch_arguments={"output_frame": frame}.items(),
        ),
        Node(
            package="clean_sweep_perception",
            executable="litter_registry",
            parameters=[{"frame": frame}],
            output="screen",
        ),
    ])
