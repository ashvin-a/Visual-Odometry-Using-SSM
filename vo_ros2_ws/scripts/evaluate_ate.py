#!/usr/bin/env python3
"""
evaluate_ate.py — Wrapper around evo_ape for Absolute Trajectory Error.

Aligns the predicted trajectory to the ground-truth using SE3 alignment
and scale correction (required for monocular VO), then prints a summary
table and saves the evo report.

Usage
-----
## Make sure your in vo_ros2_ws directory before running
python scripts/evaluate_ate.py \
    --gt   install/data/groundtruth.txt \
    --pred results/predicted_trajectory.txt \
    --out  results/evo_report
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_evo(gt: Path, pred: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        'evo_ape', 'tum',
        str(gt), str(pred),
        '--align',
        '--correct_scale',
        '--plot',
        '--plot_mode', 'xy',
        '--save_results', str(out / 'ape_results.zip'),
        '--save_plot',   str(out / 'trajectory_overlay.pdf'),
        '--verbose',
    ]

    print('Running: ' + ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print('\nevo_ape exited with non-zero status. Check the output above.', file=sys.stderr)
        sys.exit(result.returncode)

    print(f'\nevo report saved to {out}/')


def main() -> None:
    parser = argparse.ArgumentParser(description='ATE evaluation using evo')
    parser.add_argument('--gt',   default="install/data/groundtruth.txt", help='Ground-truth TUM file')
    parser.add_argument('--pred', default="../results/predicted_trajectory.txt", help='Predicted trajectory TUM file')
    parser.add_argument('--out',  default='../results/evo_report', help='Output directory')
    args = parser.parse_args()

    gt   = Path(args.gt)
    pred = Path(args.pred)
    out  = Path(args.out)

    if not gt.exists():
        print(f'Ground-truth file not found: {gt}', file=sys.stderr)
        sys.exit(1)
    if not pred.exists():
        print(f'Predicted trajectory file not found: {pred}', file=sys.stderr)
        sys.exit(1)

    run_evo(gt, pred, out)


if __name__ == '__main__':
    main()
