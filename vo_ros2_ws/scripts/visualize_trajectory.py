#!/usr/bin/env python3
"""
visualize_trajectory.py — Plot predicted vs. ground-truth trajectory.

Reads two TUM-format trajectory files and produces a 2D (X-Y plane) overlay
plot and a 3D trajectory plot.

Usage
-----
python scripts/visualize_trajectory.py \
    --gt   data/groundtruth.txt \
    --pred results/predicted_trajectory.txt \
    --out  results/trajectory_plot.png
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
    poses      : (N, 7) float64  [tx ty tz qx qy qz qw]
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


def correct_scale(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Apply the best-fit scale factor to align pred onto gt (translation only)."""
    s = (np.linalg.norm(gt_xyz) / np.linalg.norm(pred_xyz)) if np.linalg.norm(pred_xyz) > 0 else 1.0
    return pred_xyz * s


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize predicted vs. ground-truth trajectory')
    parser.add_argument('--gt',   required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out',  default='results/trajectory_plot.png')
    parser.add_argument('--correct_scale', action='store_true',
                        help='Apply naive scale correction before plotting')
    args = parser.parse_args()

    gt_path   = Path(args.gt)
    pred_path = Path(args.pred)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        print(f'Ground-truth not found: {gt_path}', file=sys.stderr); sys.exit(1)
    if not pred_path.exists():
        print(f'Predicted trajectory not found: {pred_path}', file=sys.stderr); sys.exit(1)

    _, gt_xyz   = load_tum(gt_path)
    _, pred_xyz = load_tum(pred_path)

    # Subsample to equal length
    n = min(len(gt_xyz), len(pred_xyz))
    gt_xyz   = gt_xyz[:n]
    pred_xyz = pred_xyz[:n]

    if args.correct_scale:
        pred_xyz = correct_scale(pred_xyz, gt_xyz)

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
