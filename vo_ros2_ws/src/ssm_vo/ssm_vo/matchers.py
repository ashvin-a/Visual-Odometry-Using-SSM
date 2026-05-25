"""
matchers.py — Pluggable feature matcher backends for baseline comparison.

All matchers expose the same interface as MambaGlueMatcher:

    match(kp0, sc0, desc0, kp1, sc1, desc1, image_size)
        → (pts0, pts1) matched pixel pairs, or (None, None) if degenerate

This lets them drop into VOInference without any other pipeline changes.
"""

import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# SuperGlue (Magic Leap, NeurIPS 2020)
# --------------------------------------------------------------------------- #

class SuperGlueMatcher:
    """
    SuperGlue wrapper.

    Requires the Magic Leap SuperGlue repo cloned alongside this project:
        git clone https://github.com/magicleap/SuperGluePretrainedNetwork superglue

    Weights (superglue_outdoor.pth or superglue_indoor.pth) must be placed in
    superglue/models/weights/ as provided by the Magic Leap repo, OR a direct
    path can be supplied via weights_path.
    """

    _SUPERGLUE_CONFIG = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
    }

    def __init__(
        self,
        weights: str = 'outdoor',
        device: torch.device = torch.device('cpu'),
        repo_path: str = 'superglue',
        min_matches: int = 20,
        confidence_threshold: float = 0.2,
    ) -> None:
        self.MIN_MATCHES = min_matches
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = self._load(weights, repo_path, device)

    def _load(self, weights: str, repo_path: str, device: torch.device):
        # Add the SuperGlue repo to the path so we can import its model class.
        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise RuntimeError(
                f"SuperGlue repo not found at '{repo}'. "
                "Clone it with:\n"
                "  git clone https://github.com/magicleap/SuperGluePretrainedNetwork superglue\n"
                "then download weights into superglue/models/weights/"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from models.superglue import SuperGlue
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import SuperGlue from '{repo}'. "
                "Make sure the repo was cloned correctly."
            ) from exc

        config = {**self._SUPERGLUE_CONFIG, 'weights': weights,
                  'match_threshold': self.confidence_threshold}
        model = SuperGlue(config).to(device).eval()
        return model

    @torch.no_grad()
    def match(
        self,
        kp0: np.ndarray, sc0: np.ndarray, desc0: np.ndarray,
        kp1: np.ndarray, sc1: np.ndarray, desc1: np.ndarray,
        image_size: tuple,
    ):
        W, H = image_size

        def _t(a):
            return torch.from_numpy(a).float().unsqueeze(0).to(self.device)

        data = {
            'keypoints0':   _t(kp0),        # (1, N, 2)
            'keypoints1':   _t(kp1),        # (1, M, 2)
            'scores0':      _t(sc0),        # (1, N)
            'scores1':      _t(sc1),        # (1, M)
            'descriptors0': _t(desc0.T),    # (1, 256, N)  SuperGlue expects (D, N)
            'descriptors1': _t(desc1.T),    # (1, 256, M)
            # Dummy image tensors — only shape matters for keypoint normalisation.
            'image0': torch.zeros(1, 1, H, W, device=self.device),
            'image1': torch.zeros(1, 1, H, W, device=self.device),
        }
        pred = self._model(data)
        matches = pred['matches0'][0].cpu().numpy()        # (N,) index into kp1
        scores  = pred['matching_scores0'][0].cpu().numpy()  # (N,) confidence

        valid = (matches > -1) & (scores > self.confidence_threshold)
        if valid.sum() < self.MIN_MATCHES:
            return None, None

        return kp0[valid], kp1[matches[valid]]


# --------------------------------------------------------------------------- #
# LightGlue (ETH Zürich, ICCV 2023)
# --------------------------------------------------------------------------- #

class LightGlueMatcher:
    """
    LightGlue wrapper.

    Install via:  pip install lightglue

    Weights are downloaded automatically on first use (~45 MB).

    Parameters
    ----------
    adaptive : bool
        If True, use LightGlue's default adaptive depth/width pruning
        (faster but variable computation per frame).
        If False, disable pruning (full depth, comparable to SuperGlue).
    """

    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        min_matches: int = 20,
        confidence_threshold: float = 0.5,
        adaptive: bool = False,
    ) -> None:
        self.MIN_MATCHES = min_matches
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = self._load(device, adaptive)

    def _load(self, device: torch.device, adaptive: bool):
        try:
            from lightglue import LightGlue
        except ImportError as exc:
            raise RuntimeError(
                "LightGlue is not installed. Run:  pip install lightglue"
            ) from exc

        # depth_confidence / width_confidence = -1 disables adaptive pruning.
        depth_conf  = 0.95 if adaptive else -1
        width_conf  = 0.99 if adaptive else -1
        model = LightGlue(
            features='superpoint',
            depth_confidence=depth_conf,
            width_confidence=width_conf,
        ).to(device).eval()
        return model

    @torch.no_grad()
    def match(
        self,
        kp0: np.ndarray, sc0: np.ndarray, desc0: np.ndarray,
        kp1: np.ndarray, sc1: np.ndarray, desc1: np.ndarray,
        image_size: tuple,
    ):
        W, H = image_size

        def _t(a):
            return torch.from_numpy(a).float().unsqueeze(0).to(self.device)

        size = torch.tensor([[W, H]], dtype=torch.float32, device=self.device)

        feats0 = {
            'keypoints':       _t(kp0),   # (1, N, 2)
            'keypoint_scores': _t(sc0),   # (1, N)
            'descriptors':     _t(desc0), # (1, N, 256)  LightGlue expects (N, D)
            'image_size':      size,
        }
        feats1 = {
            'keypoints':       _t(kp1),
            'keypoint_scores': _t(sc1),
            'descriptors':     _t(desc1),
            'image_size':      size,
        }
        pred = self._model({'image0': feats0, 'image1': feats1})
        matches = pred['matches0'][0].cpu().numpy()        # (N,) index into kp1
        scores  = pred['matching_scores0'][0].cpu().numpy()  # (N,) confidence

        valid = (matches > -1) & (scores > self.confidence_threshold)
        if valid.sum() < self.MIN_MATCHES:
            return None, None

        return kp0[valid], kp1[matches[valid]]
