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
import bisect
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'ssm_vo'))

from ssm_vo.inference import VOInference, MambaGlueMatcher
from ssm_vo.matchers import SuperGlueMatcher, LightGlueMatcher
from ssm_vo.pose_estimator import TrajectoryAccumulator


def _load_gt(gt_file: Path):
    """Load a TUM-format ground truth file → (timestamps list, positions dict)."""
    timestamps = []
    positions = {}
    with open(gt_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            timestamps.append(ts)
            positions[ts] = pos
    timestamps.sort()
    return timestamps, positions


def _interp_position(timestamps, positions, ts: float):
    """Linearly interpolate position at timestamp ts."""
    if not timestamps:
        return None
    if ts <= timestamps[0]:
        return positions[timestamps[0]]
    if ts >= timestamps[-1]:
        return positions[timestamps[-1]]
    idx = bisect.bisect_left(timestamps, ts)
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    alpha = (ts - t0) / (t1 - t0)
    return positions[t0] + alpha * (positions[t1] - positions[t0])


def _gt_scale(timestamps, positions, ts_prev: float, ts_curr: float) -> float:
    """Return the GT displacement (metres) between two timestamps."""
    p0 = _interp_position(timestamps, positions, ts_prev)
    p1 = _interp_position(timestamps, positions, ts_curr)
    if p0 is None or p1 is None:
        return 1.0
    dist = float(np.linalg.norm(p1 - p0))
    return dist if dist > 1e-6 else 0.0  # 0.0 = stationary, don't move


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
    # Optional GT-scale assistance: load ground truth once up front.
    gt_timestamps, gt_positions = None, None
    if args.gt_file:
        gt_path = Path(args.gt_file)
        if not gt_path.exists():
            print(f'Warning: --gt_file {gt_path} not found — running without scale correction.',
                  file=sys.stderr)
        else:
            gt_timestamps, gt_positions = _load_gt(gt_path)
            print(f'Loaded {len(gt_timestamps)} GT poses from {gt_path} (scale-assisted mode)')

    image_dir = Path(args.data_dir)
    all_frames = sorted(image_dir.glob('*.png'), key=lambda p: parse_timestamp(p))

    # Filter to [start_ts, end_ts] window so offline runs match the GT range
    frames = [f for f in all_frames
              if args.start_ts <= parse_timestamp(f) <= args.end_ts]

    # Optionally thin the frame sequence (larger inter-frame baseline)
    if args.frame_skip > 1:
        frames = frames[::args.frame_skip]

    # Optionally cap total frames processed
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    if len(frames) < 2:
        print(f'Need at least 2 PNG images in {image_dir} within '
              f'[{args.start_ts}, {args.end_ts}], found {len(frames)}.',
              file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(all_frames)} frames total, {len(frames)} in '
          f'[{args.start_ts}, {args.end_ts}]'
          f'{f" (skip={args.frame_skip})" if args.frame_skip > 1 else ""}'
          f'{f" (capped at {args.max_frames})" if args.max_frames > 0 else ""}')

    import torch
    _device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.matcher == 'superglue':
        matcher = SuperGlueMatcher(
            weights=args.sg_weights,
            device=_device,
            repo_path=args.sg_repo,
            min_matches=args.min_matches,
            confidence_threshold=args.confidence,
        )
        print(f'Matcher: SuperGlue (weights={args.sg_weights})')
    elif args.matcher == 'lightglue':
        matcher = LightGlueMatcher(
            device=_device,
            min_matches=args.min_matches,
            confidence_threshold=args.confidence,
            adaptive=args.lg_adaptive,
        )
        mode = 'adaptive' if args.lg_adaptive else 'full-depth'
        print(f'Matcher: LightGlue ({mode})')
    else:
        matcher = None  # VOInference builds MambaGlueMatcher internally
        print(f'Matcher: MambaGlue (weights={args.mg_weights})')

    vo = VOInference(
        superpoint_weights=args.sp_weights,
        mambaglue_weights=args.mg_weights,
        camera_matrix=DEFAULT_K,
        device=args.device,
        nms_radius=args.nms_radius,
        max_keypoints=args.max_keypoints,
        keypoint_threshold=args.kp_threshold,
        min_matches=args.min_matches,
        confidence_threshold=args.confidence,
        min_inliers=args.min_inliers,
        matcher=matcher,
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

            # Apply metric scale when GT is available.
            # cv2.recoverPose always returns |t|=1 (unit-norm); multiplying by the
            # GT inter-frame displacement restores metric scale without changing shape.
            if T_rel is not None and gt_timestamps is not None:
                scale = _gt_scale(gt_timestamps, gt_positions, prev_ts, ts)
                T_rel = T_rel.copy()
                T_rel[:3, 3] *= scale

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
    parser.add_argument('--gt_file',    default=None,
                        help='TUM ground-truth file for metric scale recovery '
                             '(GT-scale-assisted mode; omit for pure monocular VO)')
    # Matcher selection
    parser.add_argument('--matcher',    default='mambaglue',
                        choices=['mambaglue', 'superglue', 'lightglue'],
                        help='Feature matcher backend (default: mambaglue)')
    parser.add_argument('--sg_weights', default='outdoor',
                        help='SuperGlue weights: "indoor", "outdoor", or path to .pth '
                             '(only used when --matcher=superglue)')
    parser.add_argument('--sg_repo',    default='superglue',
                        help='Path to cloned SuperGluePretrainedNetwork repo '
                             '(only used when --matcher=superglue)')
    parser.add_argument('--lg_adaptive', action='store_true',
                        help='Enable LightGlue adaptive depth/width pruning '
                             '(faster but variable; default: full-depth for fair comparison)')
    parser.add_argument('--start_ts',   type=float, default=0.0,
                        help='Only process frames with timestamp >= this value (seconds)')
    parser.add_argument('--end_ts',     type=float, default=float('inf'),
                        help='Only process frames with timestamp <= this value (seconds)')
    parser.add_argument('--max_dt',     type=float, default=0.5,
                        help='Skip frame pairs with timestamp gap > this value (seconds); '
                             'catches cross-session boundaries in multi-session image dirs')
    parser.add_argument('--frame_skip', type=int,   default=2,
                        help='Process every Nth frame (e.g. 2 = every other frame, larger baseline)')
    parser.add_argument('--max_frames', type=int,   default=0,
                        help='Cap the number of frames processed (0 = no cap, process all)')
    # SuperPoint hyperparameters
    parser.add_argument('--nms_radius',   type=int,   default=4,
                        help='SuperPoint NMS radius (larger = more spread keypoints)')
    parser.add_argument('--max_keypoints', type=int,  default=2048,
                        help='Maximum SuperPoint keypoints per frame')
    parser.add_argument('--kp_threshold',  type=float, default=0.0005,
                        help='SuperPoint keypoint score threshold')
    # MambaGlue hyperparameters
    parser.add_argument('--min_matches',  type=int,   default=20,
                        help='Minimum high-confidence matches required (else frame is dropped)')
    parser.add_argument('--confidence',   type=float, default=0.5,
                        help='MambaGlue match confidence threshold [0, 1]')
    # Geometry hyperparameters
    parser.add_argument('--min_inliers',  type=int,   default=8,
                        help='Minimum RANSAC inliers required after Essential Matrix estimation')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
