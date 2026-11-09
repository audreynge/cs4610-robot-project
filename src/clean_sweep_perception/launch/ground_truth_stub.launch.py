import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_objects = os.path.join(
        get_package_share_directory("clean_sweep_perception"), "config", "sim_objects.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument("objects_file", default_value=default_objects),
        Node(
            package="clean_sweep_perception",
            executable="ground_truth_stub",
            parameters=[{"objects_file": LaunchConfiguration("objects_file")}],
            output="screen",
        ),
    ])
