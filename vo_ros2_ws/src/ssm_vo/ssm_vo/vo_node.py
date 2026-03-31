"""
vo_node.py — ROS2 node: /camera/image_raw → /vo/odometry

Subscribes to the Gazebo camera topic, runs SuperPoint + MambaGlue inference
on consecutive frame pairs, accumulates the pose, and publishes odometry.

Published topics
----------------
/vo/odometry   nav_msgs/Odometry      — cumulative world-frame pose
/vo/latency    std_msgs/Float64       — per-frame wall-clock latency (ms)

Parameters
----------
superpoint_weights  : str  — path to superpoint.pth
mambaglue_weights   : str  — path to mambaglue_checkpoint_best.tar
device              : str  — 'cuda' or 'cpu'
fx, fy, cx, cy      : float — camera intrinsics
traj_output_path    : str  — path to write TUM trajectory file
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Quaternion, Point, Pose, PoseWithCovariance, Twist, TwistWithCovariance
from cv_bridge import CvBridge

from .inference import VOInference
from .pose_estimator import TrajectoryAccumulator
from .profiler import HardwareProfiler


class VONode(Node):

    def __init__(self) -> None:
        super().__init__('vo_node')

        # --- Parameters ----------------------------------------------------- #
        self.declare_parameter('superpoint_weights', 'models/superpoint.pth')
        self.declare_parameter('mambaglue_weights',  'models/mambaglue_checkpoint_best.tar')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('fx', 554.254)
        self.declare_parameter('fy', 554.254)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('traj_output_path', 'results/predicted_trajectory.txt')

        sp_w  = self.get_parameter('superpoint_weights').value
        mg_w  = self.get_parameter('mambaglue_weights').value
        dev   = self.get_parameter('device').value
        fx    = self.get_parameter('fx').value
        fy    = self.get_parameter('fy').value
        cx    = self.get_parameter('cx').value
        cy    = self.get_parameter('cy').value
        traj  = self.get_parameter('traj_output_path').value

        # --- Camera intrinsics ---------------------------------------------- #
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # --- Inference pipeline --------------------------------------------- #
        self.get_logger().info(f'Loading SuperPoint from {sp_w}')
        self.get_logger().info(f'Loading MambaGlue from  {mg_w}')
        self.vo = VOInference(sp_w, mg_w, self.K, device=dev)

        # --- State ---------------------------------------------------------- #
        self.bridge = CvBridge()
        self.prev_frame: np.ndarray | None = None
        self.accumulator = TrajectoryAccumulator()
        self.profiler = HardwareProfiler()
        self.profiler.start()

        # Trajectory output file
        traj_path = Path(traj)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        self._traj_fh = open(traj_path, 'w')

        # --- Publishers ----------------------------------------------------- #
        self._pub_odom    = self.create_publisher(Odometry, '/vo/odometry', 10)
        self._pub_latency = self.create_publisher(Float64,  '/vo/latency',  10)

        # --- Subscriber ----------------------------------------------------- #
        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)

        self.get_logger().info('VO node ready — waiting for images on /camera/image_raw')

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def _image_cb(self, msg: Image) -> None:
        t_start = time.perf_counter()
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ROS2 Image → OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.prev_frame is None:
            self.prev_frame = frame
            return

        # Run inference
        T_rel = self.vo.estimate_pose(self.prev_frame, frame)
        self.prev_frame = frame

        # Accumulate pose
        T_world = self.accumulator.update(T_rel)

        # Compute latency
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Publish odometry
        self._publish_odometry(T_world, msg.header)

        # Publish latency
        lat_msg = Float64()
        lat_msg.data = latency_ms
        self._pub_latency.publish(lat_msg)

        # Write trajectory line
        line = self.accumulator.as_tum_line(timestamp)
        self._traj_fh.write(line + '\n')
        self._traj_fh.flush()

        # Log every 30 frames
        if self.accumulator.total_frames % 30 == 0:
            timings = self.vo.timings
            self.get_logger().info(
                f'Frame {self.accumulator.total_frames} | '
                f'Latency: {latency_ms:.1f} ms | '
                f'SP: {timings.get("superpoint_ms", 0):.1f} ms | '
                f'MG: {timings.get("mambaglue_ms", 0):.1f} ms | '
                f'Geo: {timings.get("geometry_ms", 0):.1f} ms | '
                f'Dropped: {self.accumulator.dropped_frames}'
            )

    def _publish_odometry(self, T: np.ndarray, header) -> None:
        msg = Odometry()
        msg.header = header
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        from scipy.spatial.transform import Rotation
        pos = T[:3, 3]
        q   = Rotation.from_matrix(T[:3, :3]).as_quat()  # (qx, qy, qz, qw)

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        self._pub_odom.publish(msg)

    # ---------------------------------------------------------------------- #
    # Cleanup
    # ---------------------------------------------------------------------- #

    def destroy_node(self) -> None:
        self.profiler.stop()
        summary = self.profiler.summary()
        self.get_logger().info(f'GPU summary: {summary}')
        self.get_logger().info(
            f'Total frames: {self.accumulator.total_frames} | '
            f'Dropped: {self.accumulator.dropped_frames} '
            f'({self.accumulator.drop_rate * 100:.1f}%)'
        )
        self._traj_fh.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
