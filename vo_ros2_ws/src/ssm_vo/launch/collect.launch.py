"""Launch data collection nodes (image saver + ground-truth pose saver)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    ws_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')

    return LaunchDescription([
        DeclareLaunchArgument('image_dir',
                              default_value=os.path.join(ws_root, 'data', 'images')),
        DeclareLaunchArgument('gt_output',
                              default_value=os.path.join(ws_root, 'data', 'groundtruth.txt')),

        Node(
            package='data_collector',
            executable='image_saver',
            name='image_saver',
            output='screen',
            parameters=[{'output_dir': LaunchConfiguration('image_dir')}],
        ),
        Node(
            package='data_collector',
            executable='gt_pose_saver',
            name='gt_pose_saver',
            output='screen',
            parameters=[{'output_file': LaunchConfiguration('gt_output')}],
        ),
    ])
