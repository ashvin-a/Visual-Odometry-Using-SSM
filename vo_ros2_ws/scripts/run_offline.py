#!/usr/bin/env python3
"""
run_offline.py — Offline visual odometry on collected image dataset.

Reads consecutive PNG frames from the data directory, runs the full
SuperPoint + MambaGlue + Essential Matrix pipeline, accumulates the
trajectory, and writes a TUM-format file for ATE evaluation.

Usage
-----
python scripts/run_offline.py \
    --data_dir  vo_ros2_ws/install/data/images \
    --sp_weights models/superpoint.pth \
    --mg_weights models/checkpoint_best.tar \
    --output     results/predicted_trajectory.txt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'ssm_vo'))

from ssm_vo.inference import VOInference
from ssm_vo.pose_estimator import TrajectoryAccumulator


DEFAULT_K = np.array(
    [[554.254, 0,       320.0],
     [0,       554.254, 240.0],
     [0,       0,       1.0  ]],
    dtype=np.float32,
)


def parse_timestamp(path: Path) -> float:
    """Extract timestamp from filename like '100.024000.png'."""
    try:
        return float(path.stem)
    except ValueError:
        return 0.0


def run(args) -> None:
    image_dir = Path(args.data_dir)
    all_frames = sorted(image_dir.glob('*.png'), key=lambda p: parse_timestamp(p))

    # Filter to [start_ts, end_ts] window so offline runs match the GT range
    frames = [f for f in all_frames
              if args.start_ts <= parse_timestamp(f) <= args.end_ts]

    if len(frames) < 2:
        print(f'Need at least 2 PNG images in {image_dir} within '
              f'[{args.start_ts}, {args.end_ts}], found {len(frames)}.',
              file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(all_frames)} frames total, {len(frames)} in '
          f'[{args.start_ts}, {args.end_ts}]')

    vo = VOInference(
        superpoint_weights=args.sp_weights,
        mambaglue_weights=args.mg_weights,
        camera_matrix=DEFAULT_K,
        device=args.device,
    )
    acc = TrajectoryAccumulator()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_frame = None
    prev_ts = None
    total_wall_ms = []

    with open(out_path, 'w') as fh:
        for i, frame_path in enumerate(frames):
            ts = parse_timestamp(frame_path)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f'  Warning: could not read {frame_path}', file=sys.stderr)
                continue

            if prev_frame is None:
                # First frame — initialise position as identity, write to file
                T_world = acc.update(None)
                line = acc.as_tum_line(ts)
                fh.write(line + '\n')
                prev_frame = frame
                prev_ts = ts
                continue

            # Skip pairs with large timestamp gaps (different recording sessions)
            if ts - prev_ts > args.max_dt:
                print(f'  Warning: gap {ts - prev_ts:.3f}s at t={ts:.3f} — skipping pair',
                      file=sys.stderr)
                prev_frame = frame
                prev_ts = ts
                continue

            t0 = time.perf_counter()
            T_rel = vo.estimate_pose(prev_frame, frame)
            wall_ms = (time.perf_counter() - t0) * 1000

            T_world = acc.update(T_rel)
            line = acc.as_tum_line(ts)
            fh.write(line + '\n')

            if T_rel is not None:
                total_wall_ms.append(wall_ms)

            prev_frame = frame
            prev_ts = ts

            if (i + 1) % 100 == 0:
                mean_ms = np.mean(total_wall_ms) if total_wall_ms else 0.0
                print(
                    f'  [{i+1}/{len(frames)}] '
                    f'dropped={acc.dropped_frames} '
                    f'mean_latency={mean_ms:.1f}ms'
                )

    # Summary
    drop_pct = 100.0 * acc.dropped_frames / acc.total_frames if acc.total_frames else 0.0
    mean_ms  = float(np.mean(total_wall_ms)) if total_wall_ms else 0.0
    fps      = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print()
    print('=' * 50)
    print(f'Frames processed : {acc.total_frames}')
    print(f'Dropped          : {acc.dropped_frames} ({drop_pct:.1f}%)')
    print(f'Mean latency     : {mean_ms:.1f} ms  ({fps:.1f} FPS)')
    print(f'Trajectory saved : {out_path}')
    print('=' * 50)
    print()
    print('Next steps:')
    print(f'  python scripts/evaluate_ate.py --gt <groundtruth.txt> --pred {out_path}')
    print(f'  python scripts/visualize_trajectory.py --gt <groundtruth.txt> --pred {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline VO trajectory runner')
    parser.add_argument('--data_dir',   default="install/data/images",
                        help='Directory of sorted PNG frames (timestamp filenames)')
    parser.add_argument('--sp_weights', default="../models/superpoint.pth",
                        help='Path to superpoint.pth')
    parser.add_argument('--mg_weights', default="../models/checkpoint_best.tar",
                        help='Path to mambaglue_checkpoint_best.tar')
    parser.add_argument('--output',     default='../results/predicted_trajectory.txt',
                        help='Output TUM trajectory file path')
    parser.add_argument('--device',     default='cuda',
                        help='PyTorch device (cuda / cpu)')
    parser.add_argument('--start_ts',   type=float, default=0.0,
                        help='Only process frames with timestamp >= this value (seconds)')
    parser.add_argument('--end_ts',     type=float, default=float('inf'),
                        help='Only process frames with timestamp <= this value (seconds)')
    parser.add_argument('--max_dt',     type=float, default=0.5,
                        help='Skip frame pairs with timestamp gap > this value (seconds); '
                             'catches cross-session boundaries in multi-session image dirs')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
