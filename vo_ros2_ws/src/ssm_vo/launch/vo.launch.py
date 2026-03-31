"""Launch the VO node with configurable model paths and camera intrinsics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    ws_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')

    return LaunchDescription([
        DeclareLaunchArgument('superpoint_weights',
                              default_value=os.path.join(ws_root, 'models', 'superpoint.pth')),
        DeclareLaunchArgument('mambaglue_weights',
                              default_value=os.path.join(ws_root, 'models', 'mambaglue_checkpoint_best.tar')),
        DeclareLaunchArgument('device',       default_value='cuda'),
        DeclareLaunchArgument('fx',           default_value='554.254'),
        DeclareLaunchArgument('fy',           default_value='554.254'),
        DeclareLaunchArgument('cx',           default_value='320.0'),
        DeclareLaunchArgument('cy',           default_value='240.0'),
        DeclareLaunchArgument('traj_output',
                              default_value=os.path.join(ws_root, 'results', 'predicted_trajectory.txt')),

        Node(
            package='ssm_vo',
            executable='vo_node',
            name='vo_node',
            output='screen',
            parameters=[{
                'superpoint_weights': LaunchConfiguration('superpoint_weights'),
                'mambaglue_weights':  LaunchConfiguration('mambaglue_weights'),
                'device':             LaunchConfiguration('device'),
                'fx':                 LaunchConfiguration('fx'),
                'fy':                 LaunchConfiguration('fy'),
                'cx':                 LaunchConfiguration('cx'),
                'cy':                 LaunchConfiguration('cy'),
                'traj_output_path':   LaunchConfiguration('traj_output'),
            }],
        ),
    ])
