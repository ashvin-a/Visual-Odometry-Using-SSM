#!/usr/bin/env python3
"""
visualize_trajectory.py — Plot predicted vs. ground-truth trajectory.

Reads two TUM-format trajectory files and produces a 2D (X-Y plane) overlay
plot and a 3D trajectory plot.

Usage
-----
python scripts/visualize_trajectory.py \
    --gt   install/data/groundtruth.txt \
    --pred ../results/predicted_trajectory.txt \
    --out  ../results/trajectory_plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a TUM trajectory file.

    Returns
    -------
    timestamps : (N,) float64
    poses      : (N, 3) float64  [tx ty tz]
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1:4]   # timestamps, (tx, ty, tz)


def align_by_timestamp(
    ts_pred: np.ndarray, xyz_pred: np.ndarray,
    ts_gt: np.ndarray,   xyz_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each predicted timestamp find the nearest GT timestamp.
    Returns (xyz_pred, xyz_gt_aligned) with one GT pose per pred pose.
    """
    idx = np.searchsorted(ts_gt, ts_pred)
    idx = np.clip(idx, 0, len(ts_gt) - 1)
    # Pick the closer of the two neighbours
    idx_prev = np.clip(idx - 1, 0, len(ts_gt) - 1)
    diff_next = np.abs(ts_gt[idx]      - ts_pred)
    diff_prev = np.abs(ts_gt[idx_prev] - ts_pred)
    idx = np.where(diff_prev < diff_next, idx_prev, idx)
    return xyz_pred, xyz_gt[idx]


def umeyama_alignment(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    correct_scale: bool = True,
) -> np.ndarray:
    """
    Align pred_xyz onto gt_xyz using the Umeyama algorithm (same as evo --align
    --correct_scale).  Returns pred_xyz transformed by the optimal
    rotation R, translation t, and (optionally) scale s:
        aligned = s * (pred @ R.T) + t
    """
    n = pred_xyz.shape[0]
    mu_pred = pred_xyz.mean(axis=0)
    mu_gt   = gt_xyz.mean(axis=0)

    pred_c = pred_xyz - mu_pred
    gt_c   = gt_xyz   - mu_gt

    var_pred = (pred_c ** 2).sum() / n

    W = (gt_c.T @ pred_c) / n          # 3x3 cross-covariance
    U, D, Vt = np.linalg.svd(W)

    # Correct reflection
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / var_pred if correct_scale and var_pred > 0 else 1.0
    t = mu_gt - s * (R @ mu_pred)

    return (s * (pred_xyz @ R.T)) + t


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize predicted vs. ground-truth trajectory')
    parser.add_argument('--gt',   default="install/data/groundtruth.txt")
    parser.add_argument('--pred', default="../results/predicted_trajectory.txt")
    parser.add_argument('--out',  default='../results/trajectory_plot.png')
    parser.add_argument('--correct_scale', action='store_true',
                        help='Apply Umeyama SE3+scale alignment before plotting (matches evo --align --correct_scale)')
    args = parser.parse_args()

    gt_path   = Path(args.gt)
    pred_path = Path(args.pred)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        print(f'Ground-truth not found: {gt_path}', file=sys.stderr); sys.exit(1)
    if not pred_path.exists():
        print(f'Predicted trajectory not found: {pred_path}', file=sys.stderr); sys.exit(1)

    ts_gt,   gt_xyz   = load_tum(gt_path)
    ts_pred, pred_xyz = load_tum(pred_path)

    # Align GT to predicted timestamps so both cover the same time span.
    pred_xyz, gt_xyz = align_by_timestamp(ts_pred, pred_xyz, ts_gt, gt_xyz)

    # Always apply SE3 alignment (rotation + translation only, scale)
    pred_xyz = umeyama_alignment(pred_xyz, gt_xyz, correct_scale=True)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D top-down (X-Y)
    ax = axes[0]
    ax.plot(gt_xyz[:, 0],   gt_xyz[:, 1],   'b-',  label='Ground truth', linewidth=1.5)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], 'r--', label='Predicted',    linewidth=1.5)
    ax.scatter([gt_xyz[0, 0]],   [gt_xyz[0, 1]],   c='blue',  s=60, zorder=5, label='GT start')
    ax.scatter([pred_xyz[0, 0]], [pred_xyz[0, 1]], c='red',   s=60, zorder=5, label='Pred start')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-down trajectory (X-Y plane)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 3D
    ax3d = fig.add_subplot(122, projection='3d', label='3d')
    fig.delaxes(axes[1])
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],   'b-',  label='Ground truth', linewidth=1.5)
    ax3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], 'r--', label='Predicted',    linewidth=1.5)
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D trajectory')
    ax3d.legend()

    plt.suptitle('Visual Odometry Trajectory: MambaGlue (SSM) vs. Ground Truth', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Saved trajectory plot to {out_path}')


if __name__ == '__main__':
    main()
