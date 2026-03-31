"""
pose_estimator.py — Trajectory accumulator.

Maintains the running world-frame pose and accumulates relative transforms
from the inference module.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryAccumulator:
    """
    Accumulates relative 4×4 poses into a world-frame trajectory.

    Usage
    -----
    acc = TrajectoryAccumulator()
    for each frame:
        T_rel = inference.estimate_pose(prev, curr)  # may be None
        pose  = acc.update(T_rel)
        # pose is always the best available world-frame pose (4×4)
    """

    def __init__(self) -> None:
        self._T_world = np.eye(4, dtype=np.float64)   # current world-frame pose
        self.dropped_frames: int = 0
        self.total_frames: int = 0

    def update(self, T_rel: np.ndarray | None) -> np.ndarray:
        """
        Parameters
        ----------
        T_rel : 4×4 relative pose, or None (degenerate frame)

        Returns
        -------
        4×4 world-frame pose
        """
        self.total_frames += 1
        if T_rel is None:
            self.dropped_frames += 1
        else:
            self._T_world = self._T_world @ T_rel
        return self._T_world.copy()

    @property
    def position(self) -> np.ndarray:
        """Current (x, y, z) position in world frame."""
        return self._T_world[:3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Current 3×3 rotation matrix."""
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
