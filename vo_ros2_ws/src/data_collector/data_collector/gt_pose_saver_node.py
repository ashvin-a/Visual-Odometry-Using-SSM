"""
gt_pose_saver_node.py — Writes /odom ground-truth poses to TUM format.

TUM format per line:
    timestamp tx ty tz qx qy qz qw
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class GTPoseSaverNode(Node):

    def __init__(self) -> None:
        super().__init__('gt_pose_saver')
        self.declare_parameter('output_file', 'data/groundtruth.txt')

        out = Path(self.get_parameter('output_file').value)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(out, 'w')
        self._count = 0

        self.create_subscription(Odometry, '/odom', self._cb, 10)
        self.get_logger().info(f'Writing ground truth to {out}')

    def _cb(self, msg: Odometry) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        line = (
            f'{t:.6f} '
            f'{p.x:.6f} {p.y:.6f} {p.z:.6f} '
            f'{o.x:.6f} {o.y:.6f} {o.z:.6f} {o.w:.6f}'
        )
        self._fh.write(line + '\n')
        self._fh.flush()
        self._count += 1

    def destroy_node(self) -> None:
        self._fh.close()
        self.get_logger().info(f'Saved {self._count} pose lines')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GTPoseSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
