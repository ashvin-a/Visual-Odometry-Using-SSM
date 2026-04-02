"""
spawn_robot.launch.py

Launches:
  1. Ignition Gazebo 6 (Fortress) with the structured indoor world
  2. robot_state_publisher (URDF → TF)
  3. ros_gz_sim create  (spawns the robot into Ignition)
  4. ros_gz_bridge      (bridges cmd_vel, odom, tf, clock, camera topics)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_robot     = get_package_share_directory('robot_description')
    pkg_ros_gz    = get_package_share_directory('ros_gz_sim')

    urdf_file  = os.path.join(pkg_robot, 'urdf',   'diffbot_camera.urdf.xacro')
    world_file = os.path.join(pkg_robot, 'worlds', 'structured_env.world')

    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # ------------------------------------------------------------------ #
        # Ignition Gazebo 6 (Fortress)
        # -r  = run simulation immediately (don't pause on start)
        # ------------------------------------------------------------------ #
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={
                'gz_args':   world_file + ' -r',
                'gz_version': '6',
            }.items(),
        ),

        # ------------------------------------------------------------------ #
        # Robot state publisher — publishes TF tree from the URDF
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
        # Spawn robot into Ignition (delayed 3 s to let the simulator start)
        # ------------------------------------------------------------------ #
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name',  'diffbot',
                        '-topic', 'robot_description',
                        '-x', '0', '-y', '0', '-z', '0.05',
                    ],
                    output='screen',
                ),
            ],
        ),

        # ------------------------------------------------------------------ #
        # ROS 2 ↔ Ignition Transport bridge
        #
        # Syntax:  <ign_topic>@<ros_type>[<ign_type>   — Ignition → ROS2
        #          <ign_topic>@<ros_type>]<ign_type>   — ROS2 → Ignition
        # ------------------------------------------------------------------ #
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            arguments=[
                # Simulation clock (Ignition → ROS2) — needed for use_sim_time
                '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                # Drive commands (ROS2 → Ignition)
                '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                # Ground-truth odometry (Ignition → ROS2) — for data collection
                '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                # TF transforms published by the diff-drive plugin
                '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
                # Camera image (Ignition → ROS2) — main VO input
                '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
        ),
    ])
