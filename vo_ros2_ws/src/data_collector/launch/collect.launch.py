"""
collect.launch.py

Launches both data-collection nodes in parallel:
  - image_saver      : saves /camera/image_raw frames to disk as PNGs
  - gt_pose_saver    : records /odom ground-truth poses in TUM format

Usage (after building the workspace):
    ros2 launch data_collector collect.launch.py
    ros2 launch data_collector collect.launch.py \\
        image_dir:=/abs/path/images \\
        gt_output:=/abs/path/groundtruth.txt
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Resolve default paths relative to the workspace root
    # Installed path: <ws>/install/data_collector/share/data_collector/launch/
    ws_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_dir',
            default_value=os.path.join(ws_root, 'data', 'images'),
            description='Directory where captured PNG frames are written',
        ),
        DeclareLaunchArgument(
            'gt_output',
            default_value=os.path.join(ws_root, 'data', 'groundtruth.txt'),
            description='File path for TUM-format ground-truth trajectory',
        ),

        Node(
            package='data_collector',
            executable='image_saver',
            name='image_saver',
            output='screen',
            parameters=[{
                'output_dir': LaunchConfiguration('image_dir'),
                'use_sim_time': True,
            }],
        ),
        Node(
            package='data_collector',
            executable='gt_pose_saver',
            name='gt_pose_saver',
            output='screen',
            parameters=[{
                'output_file': LaunchConfiguration('gt_output'),
                'use_sim_time': True,
            }],
        ),
    ])
