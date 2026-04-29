"""
pose_estimator.py — Trajectory accumulator.

Maintains the running world-frame pose and accumulates relative transforms
from the inference module.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Rotation from camera optical frame to robot body frame.
# cv2.recoverPose returns R,t in camera optical convention (Z=forward, X=right, Y=down).
# Derived from URDF rpy=(-π/2, 0, -π/2): R = Rz(-π/2) @ Rx(-π/2).
# Maps: Z_cam→X_robot, X_cam→-Y_robot, Y_cam→-Z_robot.
# Without this transform, forward motion accumulates on the Z axis instead of X.
_R_RC = np.array([
    [ 0.,  0.,  1.],
    [-1.,  0.,  0.],
    [ 0., -1.,  0.],
], dtype=np.float64)
_T_RC = np.eye(4, dtype=np.float64)
_T_RC[:3, :3] = _R_RC


def _enforce_planar(T: np.ndarray) -> np.ndarray:
    """Zero out Z translation and roll/pitch — keep only X, Y, yaw.

    A differential-drive robot on flat ground has exactly 3 DoF. Any roll,
    pitch, or Z translation in T_rel is measurement noise; accumulating it
    compounds Z-drift over hundreds of frames.
    """
    T_out = T.copy()
    T_out[2, 3] = 0.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    rpy[0] = 0.0  # zero roll
    rpy[1] = 0.0  # zero pitch
    T_out[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return T_out


class TrajectoryAccumulator:
    """
    Accumulates relative 4x4 poses into a world-frame trajectory.

    Usage
    -----
    acc = TrajectoryAccumulator()
    for each frame:
        T_rel = inference.estimate_pose(prev, curr)  # may be None
        pose  = acc.update(T_rel)
        # pose is always the best available world-frame pose (4x4)
    """

    def __init__(self) -> None:
        self._T_world = np.eye(4, dtype=np.float64)   # current world-frame pose
        self.dropped_frames: int = 0
        self.total_frames: int = 0

    def update(self, T_rel: np.ndarray | None) -> np.ndarray:
        """
        Parameters
        ----------
        T_rel : 4x4 relative pose in camera optical frame, or None (degenerate)

        Returns
        -------
        4x4 world-frame pose
        """
        self.total_frames += 1
        if T_rel is None:
            self.dropped_frames += 1
        else:
            # Re-express T_rel in robot body frame, then enforce planarity
            T_rel_robot = _T_RC @ T_rel @ _T_RC.T
            T_rel_robot = _enforce_planar(T_rel_robot)
            self._T_world = self._T_world @ T_rel_robot
        return self._T_world.copy()

    @property
    def position(self) -> np.ndarray:
        """Current (x, y, z) position in world frame."""
        return self._T_world[:3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Current 3x3 rotation matrix."""
        return self._T_world[:3, :3]

    @property
    def quaternion(self) -> np.ndarray:
        """Current orientation as (qx, qy, qz, qw)."""
        return Rotation.from_matrix(self.rotation_matrix).as_quat()

    def as_tum_line(self, timestamp: float) -> str:
        """
        Format current pose as a TUM trajectory line:
            timestamp tx ty tz qx qy qz qw
        """
        tx, ty, tz = self.position
        qx, qy, qz, qw = self.quaternion
        return f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"

    @property
    def drop_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.dropped_frames / self.total_frames
