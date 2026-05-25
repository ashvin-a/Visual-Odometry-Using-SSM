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
from ssm_vo.matchers import SuperGlueMatcher, LightGlueMatcher
from ssm_vo.profiler import HardwareProfiler


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

    import torch
    _device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.matcher == 'superglue':
        matcher = SuperGlueMatcher(
            weights=args.sg_weights,
            device=_device,
            repo_path=args.sg_repo,
        )
        print(f'Matcher: SuperGlue (weights={args.sg_weights})')
    elif args.matcher == 'lightglue':
        matcher = LightGlueMatcher(
            device=_device,
            adaptive=args.lg_adaptive,
        )
        mode = 'adaptive' if args.lg_adaptive else 'full-depth'
        print(f'Matcher: LightGlue ({mode})')
    else:
        matcher = None
        print(f'Matcher: MambaGlue (weights={args.mg_weights})')

    vo = VOInference(
        superpoint_weights=args.sp_weights,
        mambaglue_weights=args.mg_weights,
        camera_matrix=DEFAULT_K,
        device=args.device,
        matcher=matcher,
    )

    # Warmup: run a few pairs so CUDA kernels compile before measuring
    WARMUP = min(5, len(pairs))
    print(f'Warming up ({WARMUP} pairs)...')
    for p0, p1 in pairs[:WARMUP]:
        f0 = cv2.imread(str(p0))
        f1 = cv2.imread(str(p1))
        if f0 is not None and f1 is not None:
            vo.estimate_pose(f0, f1)

    latencies   = []
    sp_times    = []
    mg_times    = []
    geo_times   = []
    dropped     = 0
    log_rows    = []

    print(f'Running benchmark on {len(pairs)} pairs (device: {args.device})...\n')

    profiler = HardwareProfiler(log_path=Path(args.output).parent / 'gpu_log.csv')
    profiler.start()

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
            mg_times.append(vo.timings.get('matcher_ms', 0))
            geo_times.append(vo.timings.get('geometry_ms', 0))

        log_rows.append({
            'pair':          i,
            'wall_ms':       round(wall_ms, 3),
            'superpoint_ms': round(vo.timings.get('superpoint_ms', 0), 3),
            'matcher_ms':    round(vo.timings.get('matcher_ms', 0), 3),
            'geometry_ms':   round(vo.timings.get('geometry_ms', 0), 3),
            'degenerate':    int(T_rel is None),
        })

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(pairs)}] last latency: {wall_ms:.1f} ms')

    profiler.stop()
    hw = profiler.summary()

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
    print(f'{"Matcher (ms)":<30} {mg["mean"]:>8.1f}  {mg["std"]:>8.1f}  {mg["p95"]:>8.1f}')
    print(f'{"Geometry (ms)":<30} {geo["mean"]:>8.1f} {geo["std"]:>8.1f} {geo["p95"]:>8.1f}')
    print()
    print(f'{"End-to-end FPS":<30} {fps:>8.1f}')
    print()
    print(f'{"GPU util mean":<30} {hw["gpu_util_mean_%"]:>7.1f}%')
    print(f'{"GPU util peak":<30} {hw["gpu_util_peak_%"]:>7.1f}%')
    print(f'{"VRAM mean":<30} {hw["vram_mean_mb"]:>7.0f} MB')
    print(f'{"VRAM peak":<30} {hw["vram_peak_mb"]:>7.0f} MB')
    print('=' * 60)

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if log_rows:
        with open(out_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f'\nPer-frame log written to {out_path}')
    else:
        print('\nNo frames processed — CSV not written.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark inference latency')
    parser.add_argument('--data_dir',   default='data/images',
                        help='Directory of sorted PNG frames')
    parser.add_argument('--sp_weights', default='models/superpoint.pth',
                        help='Path to superpoint.pth')
    parser.add_argument('--mg_weights', default='models/checkpoint_best.tar',
                        help='Path to mambaglue checkpoint_best.tar')
    parser.add_argument('--n_pairs',    type=int, default=500,
                        help='Number of consecutive frame pairs to evaluate')
    parser.add_argument('--device',     default='cuda',
                        help='PyTorch device string (cuda / cpu)')
    parser.add_argument('--output',     default='results/latency_log.csv',
                        help='Output CSV path')
    parser.add_argument('--matcher',    default='mambaglue',
                        choices=['mambaglue', 'superglue', 'lightglue'],
                        help='Feature matcher backend (default: mambaglue)')
    parser.add_argument('--sg_weights', default='outdoor',
                        help='SuperGlue weights: "indoor", "outdoor", or path to .pth')
    parser.add_argument('--sg_repo',    default='superglue',
                        help='Path to cloned SuperGluePretrainedNetwork repo')
    parser.add_argument('--lg_adaptive', action='store_true',
                        help='Enable LightGlue adaptive depth/width pruning')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
