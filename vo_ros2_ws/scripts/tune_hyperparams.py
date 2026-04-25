#!/usr/bin/env python3
"""
tune_hyperparams.py — One-at-a-time hyperparameter sweep for the VO pipeline.

For each config, runs the inference pipeline on a fixed subset of the dataset,
evaluates ATE RMSE using evo's Python API, and prints a ranked results table.

Usage
-----
# From vo_ros2_ws/:
python scripts/tune_hyperparams.py \
    --data_dir  install/data/images \
    --gt        install/data/groundtruth.txt \
    --sp_weights ../models/superpoint.pth \
    --mg_weights ../models/checkpoint_best.tar \
    --n_frames  600        # frames per config (0 = all)
    --device    cuda
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'ssm_vo'))

from ssm_vo.inference import VOInference
from ssm_vo.pose_estimator import TrajectoryAccumulator

try:
    import copy
    from evo.tools import file_interface
    from evo.core import metrics, sync, geometry, lie_algebra as lie
    from evo.core.metrics import APE, PoseRelation
    _EVO_OK = True
except ImportError:
    _EVO_OK = False

DEFAULT_K = np.array(
    [[554.254, 0,       320.0],
     [0,       554.254, 240.0],
     [0,       0,       1.0  ]],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Experiment definitions — one variable changed at a time from baseline
# ---------------------------------------------------------------------------

BASELINE = dict(
    nms_radius=4,
    max_keypoints=2048,
    kp_threshold=0.0005,
    min_matches=20,
    confidence=0.5,
    min_inliers=8,
    frame_skip=1,
)

EXPERIMENTS = [
    # ── Baseline ────────────────────────────────────────────────────────────
    {'name': 'baseline',                **BASELINE},

    # ── RANSAC inlier minimum ───────────────────────────────────────────────
    {'name': 'min_inliers=12',          **{**BASELINE, 'min_inliers': 12}},
    {'name': 'min_inliers=15',          **{**BASELINE, 'min_inliers': 15}},
    {'name': 'min_inliers=20',          **{**BASELINE, 'min_inliers': 20}},

    # ── SuperPoint keypoint threshold ───────────────────────────────────────
    {'name': 'kp_threshold=0.001',      **{**BASELINE, 'kp_threshold': 0.001}},
    {'name': 'kp_threshold=0.005',      **{**BASELINE, 'kp_threshold': 0.005}},

    # ── Max keypoints ───────────────────────────────────────────────────────
    {'name': 'max_keypoints=1024',      **{**BASELINE, 'max_keypoints': 1024}},
    {'name': 'max_keypoints=512',       **{**BASELINE, 'max_keypoints': 512}},

    # ── MambaGlue confidence threshold ─────────────────────────────────────
    {'name': 'confidence=0.6',          **{**BASELINE, 'confidence': 0.6}},
    {'name': 'confidence=0.7',          **{**BASELINE, 'confidence': 0.7}},

    # ── NMS radius ──────────────────────────────────────────────────────────
    {'name': 'nms_radius=6',            **{**BASELINE, 'nms_radius': 6}},
    {'name': 'nms_radius=8',            **{**BASELINE, 'nms_radius': 8}},

    # ── Frame skip (larger inter-frame baseline) ────────────────────────────
    {'name': 'frame_skip=2',            **{**BASELINE, 'frame_skip': 2}},
    {'name': 'frame_skip=3',            **{**BASELINE, 'frame_skip': 3}},

    # ── Combined best guesses (add after initial sweep informs choices) ─────
    {'name': 'combo_tight',             **{**BASELINE,
                                           'min_inliers': 15,
                                           'confidence': 0.6,
                                           'kp_threshold': 0.001}},
]


def parse_timestamp(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError:
        return 0.0


def load_frames(image_dir: Path, n_frames: int, frame_skip: int) -> list:
    all_frames = sorted(image_dir.glob('*.png'), key=parse_timestamp)
    frames = all_frames[::frame_skip]
    if n_frames > 0:
        frames = frames[:n_frames]
    return frames


def run_config(cfg: dict, frames: list, sp_weights: str, mg_weights: str,
               device: str, max_dt: float) -> tuple[list[str], dict]:
    """Run the VO pipeline for one hyperparameter config.

    Returns (tum_lines, stats) where stats contains timing and drop-rate info.
    """
    vo = VOInference(
        superpoint_weights=sp_weights,
        mambaglue_weights=mg_weights,
        camera_matrix=DEFAULT_K,
        device=device,
        nms_radius=cfg['nms_radius'],
        max_keypoints=cfg['max_keypoints'],
        keypoint_threshold=cfg['kp_threshold'],
        min_matches=cfg['min_matches'],
        confidence_threshold=cfg['confidence'],
        min_inliers=cfg['min_inliers'],
    )
    acc = TrajectoryAccumulator()
    tum_lines: list[str] = []
    wall_ms: list[float] = []

    prev_frame = None
    prev_ts = None

    for frame_path in frames:
        ts = parse_timestamp(frame_path)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        if prev_frame is None:
            acc.update(None)
            tum_lines.append(acc.as_tum_line(ts))
            prev_frame = frame
            prev_ts = ts
            continue

        if ts - prev_ts > max_dt:
            prev_frame = frame
            prev_ts = ts
            continue

        t0 = time.perf_counter()
        T_rel = vo.estimate_pose(prev_frame, frame)
        elapsed = (time.perf_counter() - t0) * 1000

        acc.update(T_rel)
        tum_lines.append(acc.as_tum_line(ts))

        if T_rel is not None:
            wall_ms.append(elapsed)

        prev_frame = frame
        prev_ts = ts

    drop_pct = 100.0 * acc.dropped_frames / acc.total_frames if acc.total_frames else 0.0
    mean_ms = float(np.mean(wall_ms)) if wall_ms else 0.0

    stats = {
        'total_frames': acc.total_frames,
        'drop_pct': drop_pct,
        'mean_ms': mean_ms,
        'fps': 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }
    return tum_lines, stats


def evaluate_rmse(tum_lines: list[str], gt_path: Path) -> float | None:
    """Write tum_lines to a temp file and compute ATE RMSE via evo."""
    if not _EVO_OK:
        return None
    if len(tum_lines) < 3:
        return None

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as fh:
        fh.write('\n'.join(tum_lines) + '\n')
        pred_path = fh.name

    try:
        traj_ref = file_interface.read_tum_trajectory_file(str(gt_path))
        traj_est = file_interface.read_tum_trajectory_file(pred_path)

        traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est)
        if len(traj_ref_s.timestamps) < 3:
            return None

        # Umeyama alignment with scale correction (evo 1.x stable API)
        R, t, s = geometry.umeyama_alignment(
            traj_est_s.positions_xyz.T,
            traj_ref_s.positions_xyz.T,
            with_scale=True,
        )
        traj_est_aligned = copy.deepcopy(traj_est_s)
        traj_est_aligned.transform(lie.sim3(R, t, s))

        ape = APE(PoseRelation.translation_part)
        ape.process_data((traj_ref_s, traj_est_aligned))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))
    except Exception as exc:
        print(f'    [evo error] {exc}')
        return None
    finally:
        Path(pred_path).unlink(missing_ok=True)


def print_table(results: list[dict]) -> None:
    sorted_results = sorted(
        [r for r in results if r['rmse'] is not None],
        key=lambda r: r['rmse'],
    )
    failed = [r for r in results if r['rmse'] is None]

    col_w = 26
    print()
    print('=' * 85)
    print(f'  {"Config":<{col_w}}  {"ATE RMSE (m)":>12}  {"Drop%":>6}  {"FPS":>6}  {"ms/pair":>8}')
    print('-' * 85)

    for i, r in enumerate(sorted_results):
        tag = ' ← best' if i == 0 else ('  ← baseline' if r['name'] == 'baseline' else '')
        print(
            f'  {r["name"]:<{col_w}}  {r["rmse"]:>12.4f}  '
            f'{r["drop_pct"]:>5.1f}%  {r["fps"]:>5.1f}  {r["mean_ms"]:>7.1f}ms'
            f'{tag}'
        )

    if failed:
        print()
        print('  Failed (too few matched timestamps):')
        for r in failed:
            print(f'    {r["name"]}')

    print('=' * 85)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for VO pipeline')
    parser.add_argument('--data_dir',    default='install/data/images')
    parser.add_argument('--gt',          default='install/data/groundtruth.txt')
    parser.add_argument('--sp_weights',  default='../models/superpoint.pth')
    parser.add_argument('--mg_weights',  default='../models/checkpoint_best.tar')
    parser.add_argument('--n_frames',    type=int,   default=600,
                        help='Frames per config (0 = all frames, much slower)')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--max_dt',      type=float, default=0.5)
    parser.add_argument('--configs',     nargs='*',
                        help='Run only these named configs (default: all)')
    args = parser.parse_args()

    if not _EVO_OK:
        print('WARNING: evo not installed — RMSE evaluation disabled.', file=sys.stderr)
        print('  pip install evo', file=sys.stderr)

    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f'Ground-truth not found: {gt_path}', file=sys.stderr)
        sys.exit(1)

    experiments = EXPERIMENTS
    if args.configs:
        experiments = [e for e in EXPERIMENTS if e['name'] in args.configs]
        if not experiments:
            print(f'No matching configs found. Available: {[e["name"] for e in EXPERIMENTS]}')
            sys.exit(1)

    image_dir = Path(args.data_dir)
    results = []

    for exp in experiments:
        frames = load_frames(image_dir, args.n_frames, exp['frame_skip'])
        if len(frames) < 2:
            print(f'\n[{exp["name"]}] Not enough frames — skipping.')
            continue

        print(f'\n[{exp["name"]}] {len(frames)} frames  '
              f'(skip={exp["frame_skip"]}, conf={exp["confidence"]}, '
              f'min_inliers={exp["min_inliers"]}, kp_thr={exp["kp_threshold"]}, '
              f'max_kp={exp["max_keypoints"]}, nms_r={exp["nms_radius"]}, '
              f'min_matches={exp["min_matches"]})')

        t_start = time.perf_counter()
        tum_lines, stats = run_config(
            exp, frames,
            sp_weights=args.sp_weights,
            mg_weights=args.mg_weights,
            device=args.device,
            max_dt=args.max_dt,
        )
        wall_total = time.perf_counter() - t_start

        print(f'  done in {wall_total:.1f}s — '
              f'drop={stats["drop_pct"]:.1f}%  fps={stats["fps"]:.1f}  '
              f'mean={stats["mean_ms"]:.1f}ms')

        rmse = evaluate_rmse(tum_lines, gt_path)
        if rmse is not None:
            print(f'  ATE RMSE = {rmse:.4f} m')
        else:
            print('  ATE RMSE = n/a (evo failed or too few timestamps)')

        results.append({'name': exp['name'], 'rmse': rmse, **stats})

    print_table(results)


if __name__ == '__main__':
    main()
