"""
spawn_robot.launch.py

Launches:
  1. Gazebo with the structured indoor world
  2. robot_state_publisher (URDF → TF)
  3. spawn_entity (drops the robot into Gazebo)
"""

import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_robot = get_package_share_directory('robot_description')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    urdf_file  = os.path.join(pkg_robot, 'urdf', 'diffbot_camera.urdf.xacro')
    world_file = os.path.join(pkg_robot, 'worlds', 'structured_env.world')

    # Process xacro → URDF string at launch time
    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([
        # ------------------------------------------------------------------ #
        # Launch arguments
        # ------------------------------------------------------------------ #
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # ------------------------------------------------------------------ #
        # Gazebo server + client
        # ------------------------------------------------------------------ #
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={
                'world': world_file,
                'verbose': 'false',
            }.items(),
        ),

        # ------------------------------------------------------------------ #
        # Robot state publisher (publishes TF from URDF joint states)
        # ------------------------------------------------------------------ #
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
        ),

        # ------------------------------------------------------------------ #
        # Spawn robot into Gazebo (delayed to let Gazebo start first)
        # ------------------------------------------------------------------ #
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', 'diffbot',
                        '-topic', 'robot_description',
                        '-x', '0', '-y', '0', '-z', '0.05',
                    ],
                    output='screen',
                ),
            ],
        ),
    ])
