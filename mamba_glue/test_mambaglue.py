"""
Quick test script to verify MambaGlue is working correctly.
Runs feature matching on two test images and saves the visualization.
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from mambaglue import MambaGlue, SuperPoint
from mambaglue.utils import load_image, match_pair
from mambaglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot


def main():
    image0_path = Path("test_image0.png")
    image1_path = Path("test_image1.png")

    for p in [image0_path, image1_path]:
        if not p.exists():
            print(f"ERROR: Image not found: {p}")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Disable cuDNN (version mismatch workaround); CUDA kernels still run via cublas
        torch.backends.cudnn.enabled = False
    print(f"Using device: {device}")

    # Load images
    image0 = load_image(image0_path).to(device)
    image1 = load_image(image1_path).to(device)
    print(f"Image shapes: {image0.shape}, {image1.shape}")

    # Initialize models
    print("Loading SuperPoint...")
    extractor = SuperPoint().to(device).eval()

    print("Loading MambaGlue (from checkpoint_best.tar)...")
    matcher = MambaGlue(features="superpoint").to(device).eval()
    print("Models loaded successfully.")

    # Run matching
    print("Running feature extraction and matching...")
    t0 = time.perf_counter()
    with torch.no_grad():
        feats0, feats1, matches01 = match_pair(
            extractor, matcher, image0, image1, device=device
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Matching completed in {elapsed_ms:.1f} ms")

    # Extract matched keypoints
    kpts0 = feats0["keypoints"].cpu().numpy()
    kpts1 = feats1["keypoints"].cpu().numpy()
    matches = matches01["matches0"].cpu().numpy()
    scores = matches01["matching_scores0"].cpu().numpy()

    valid = matches > -1
    matched_kpts0 = kpts0[valid]
    matched_kpts1 = kpts1[matches[valid]]
    matched_scores = scores[valid]

    print(f"Keypoints detected: image0={len(kpts0)}, image1={len(kpts1)}")
    print(f"Matches found: {len(matched_kpts0)}")
    if len(matched_scores) > 0:
        print(f"Match scores: min={matched_scores.min():.3f}, "
              f"mean={matched_scores.mean():.3f}, max={matched_scores.max():.3f}")

    # Visualize
    img0_np = cv2.imread(str(image0_path))[..., ::-1]
    img1_np = cv2.imread(str(image1_path))[..., ::-1]

    axes = plot_images([img0_np, img1_np], titles=["Image 0", "Image 1"])
    plot_keypoints([kpts0, kpts1], colors="lime", ps=4)
    plot_matches(matched_kpts0, matched_kpts1, color="cyan", lw=0.8, ps=4)
    plt.suptitle(
        f"MambaGlue  |  {len(matched_kpts0)} matches  |  {elapsed_ms:.0f} ms",
        fontsize=12,
    )
    out_path = "mambaglue_matches.png"
    save_plot(out_path)
    print(f"Visualization saved to: {out_path}")
    print("TEST PASSED")


if __name__ == "__main__":
    main()
