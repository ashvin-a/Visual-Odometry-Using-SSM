"""
image_saver_node.py — Saves /camera/image_raw frames to disk.

Each frame is saved as `{timestamp:.6f}.png` in the output directory.
"""

import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSaverNode(Node):

    def __init__(self) -> None:
        super().__init__('image_saver')
        self.declare_parameter('output_dir', 'data/images')

        out = Path(self.get_parameter('output_dir').value)
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = out

        self.bridge = CvBridge()
        self._count = 0

        self.create_subscription(Image, '/camera/image_raw', self._cb, 10)
        self.get_logger().info(f'Saving images to {self.output_dir}')

    def _cb(self, msg: Image) -> None:
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = self.output_dir / f'{timestamp:.6f}.png'
        cv2.imwrite(str(filename), frame)
        self._count += 1
        if self._count % 100 == 0:
            self.get_logger().info(f'Saved {self._count} frames')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Total frames saved: {node._count}')
        node.destroy_node()
        rclpy.shutdown()
