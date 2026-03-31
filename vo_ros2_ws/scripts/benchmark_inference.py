#!/usr/bin/env python3
"""
benchmark_inference.py — Standalone latency/FPS benchmark (no ROS needed).

Runs N consecutive frame pairs from the collected dataset through the full
SuperPoint + MambaGlue + Essential Matrix pipeline and reports latency stats.

Usage
-----
python scripts/benchmark_inference.py \
    --data_dir  data/images \
    --sp_weights models/superpoint.pth \
    --mg_weights models/mambaglue_checkpoint_best.tar \
    --n_pairs 500
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow importing from the ssm_vo source tree without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'ssm_vo'))

from ssm_vo.inference import VOInference


# --------------------------------------------------------------------------- #
# Default Gazebo intrinsics (640×480, 80° FOV horizontal)
# fx = fy = W / (2 * tan(fov/2)) = 640 / (2 * tan(40°)) ≈ 554.26
# --------------------------------------------------------------------------- #
DEFAULT_K = np.array(
    [[554.254, 0,       320.0],
     [0,       554.254, 240.0],
     [0,       0,       1.0  ]],
    dtype=np.float32,
)


def collect_frame_paths(image_dir: Path, n_pairs: int) -> list[tuple[Path, Path]]:
    """Return sorted consecutive (frame0, frame1) path pairs."""
    files = sorted(image_dir.glob('*.png'))
    if len(files) < 2:
        raise FileNotFoundError(
            f'Need at least 2 PNG images in {image_dir}, found {len(files)}.'
        )
    pairs = [(files[i], files[i + 1]) for i in range(min(n_pairs, len(files) - 1))]
    return pairs


def run(args) -> None:
    image_dir = Path(args.data_dir)
    pairs = collect_frame_paths(image_dir, args.n_pairs)
    print(f'Loaded {len(pairs)} frame pairs from {image_dir}')

    vo = VOInference(
        superpoint_weights=args.sp_weights,
        mambaglue_weights=args.mg_weights,
        camera_matrix=DEFAULT_K,
        device=args.device,
    )

    latencies   = []
    sp_times    = []
    mg_times    = []
    geo_times   = []
    dropped     = 0
    log_rows    = []

    print(f'Running benchmark on {len(pairs)} pairs (device: {args.device})...\n')

    for i, (p0, p1) in enumerate(pairs):
        f0 = cv2.imread(str(p0))
        f1 = cv2.imread(str(p1))
        if f0 is None or f1 is None:
            dropped += 1
            continue

        t_start = time.perf_counter()
        T_rel = vo.estimate_pose(f0, f1)
        wall_ms = (time.perf_counter() - t_start) * 1000

        if T_rel is None:
            dropped += 1
        else:
            latencies.append(wall_ms)
            sp_times.append(vo.timings.get('superpoint_ms', 0))
            mg_times.append(vo.timings.get('mambaglue_ms', 0))
            geo_times.append(vo.timings.get('geometry_ms', 0))

        log_rows.append({
            'pair':          i,
            'wall_ms':       round(wall_ms, 3),
            'superpoint_ms': round(vo.timings.get('superpoint_ms', 0), 3),
            'mambaglue_ms':  round(vo.timings.get('mambaglue_ms', 0), 3),
            'geometry_ms':   round(vo.timings.get('geometry_ms', 0), 3),
            'degenerate':    int(T_rel is None),
        })

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(pairs)}] last latency: {wall_ms:.1f} ms')

    # ----------------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------------- #
    def _stats(data: list[float]) -> dict:
        if not data:
            return {'mean': 0, 'std': 0, 'p95': 0}
        arr = np.array(data)
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'p95':  float(np.percentile(arr, 95)),
        }

    lat  = _stats(latencies)
    sp   = _stats(sp_times)
    mg   = _stats(mg_times)
    geo  = _stats(geo_times)
    fps  = 1000.0 / lat['mean'] if lat['mean'] > 0 else 0.0
    drop_pct = 100.0 * dropped / len(pairs) if pairs else 0.0

    print('\n' + '=' * 60)
    print('BENCHMARK RESULTS')
    print('=' * 60)
    print(f'Frame pairs evaluated : {len(pairs)}')
    print(f'Dropped (degenerate)  : {dropped} ({drop_pct:.1f}%)')
    print()
    print(f'{"Metric":<30} {"Mean":>8} {"Std":>8} {"P95":>8}')
    print('-' * 56)
    print(f'{"Total latency (ms)":<30} {lat["mean"]:>8.1f} {lat["std"]:>8.1f} {lat["p95"]:>8.1f}')
    print(f'{"SuperPoint (ms)":<30} {sp["mean"]:>8.1f}  {sp["std"]:>8.1f}  {sp["p95"]:>8.1f}')
    print(f'{"MambaGlue (ms)":<30} {mg["mean"]:>8.1f}  {mg["std"]:>8.1f}  {mg["p95"]:>8.1f}')
    print(f'{"Geometry (ms)":<30} {geo["mean"]:>8.1f} {geo["std"]:>8.1f} {geo["p95"]:>8.1f}')
    print()
    print(f'{"End-to-end FPS":<30} {fps:>8.1f}')
    print('=' * 60)

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f'\nPer-frame log written to {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark inference latency')
    parser.add_argument('--data_dir',   default='data/images',
                        help='Directory of sorted PNG frames')
    parser.add_argument('--sp_weights', default='models/superpoint.pth',
                        help='Path to superpoint.pth')
    parser.add_argument('--mg_weights', default='models/mambaglue_checkpoint_best.tar',
                        help='Path to mambaglue_checkpoint_best.tar')
    parser.add_argument('--n_pairs',    type=int, default=500,
                        help='Number of consecutive frame pairs to evaluate')
    parser.add_argument('--device',     default='cuda',
                        help='PyTorch device string (cuda / cpu)')
    parser.add_argument('--output',     default='results/latency_log.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
