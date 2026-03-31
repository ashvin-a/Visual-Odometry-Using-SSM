"""
inference.py — Standalone inference module (no ROS dependency).

Takes two consecutive frames and returns a 4×4 relative pose matrix.
Uses SuperPoint for keypoint detection and MambaGlue for feature matching.
"""

import time
import numpy as np
import cv2
import torch

# --------------------------------------------------------------------------- #
# SuperPoint implementation (self-contained, no external repo needed)
# Based on the MagicLeap SuperPoint reference implementation.
# --------------------------------------------------------------------------- #

def _nms(heatmap: np.ndarray, radius: int = 4) -> np.ndarray:
    """Non-maximum suppression on a keypoint heatmap."""
    from scipy.ndimage import maximum_filter
    pad = radius
    local_max = maximum_filter(heatmap, size=2 * radius + 1)
    return (heatmap == local_max) & (heatmap > 0)


class SuperPointNet(torch.nn.Module):
    """SuperPoint network (encoder + keypoint/descriptor heads)."""

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, d = 64, 64, 128, 128, 256

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Shared encoder
        self.conv1a = torch.nn.Conv2d(1, c1, 3, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, 3, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, 3, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, 3, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, 3, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, 3, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, 3, padding=1)

        # Keypoint detector head
        self.convPa = torch.nn.Conv2d(c4, 256, 3, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, 1)

        # Descriptor head
        self.convDa = torch.nn.Conv2d(c4, 256, 3, padding=1)
        self.convDb = torch.nn.Conv2d(256, d, 1)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Keypoint head
        kp = self.relu(self.convPa(x))
        kp = self.convPb(kp)  # (B, 65, H/8, W/8)
        kp = torch.nn.functional.softmax(kp, dim=1)

        # Descriptor head
        desc = self.relu(self.convDa(x))
        desc = self.convDb(desc)  # (B, 256, H/8, W/8)
        desc = torch.nn.functional.normalize(desc, dim=1)

        return kp, desc


class SuperPoint:
    """SuperPoint keypoint detector and descriptor extractor."""

    TARGET_W = 640
    TARGET_H = 480
    NMS_RADIUS = 4
    MAX_KEYPOINTS = 1024
    KEYPOINT_THRESHOLD = 0.005

    def __init__(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        self.net = SuperPointNet().to(device)
        ckpt = torch.load(weights_path, map_location=device)
        # Weights may be stored directly or under a 'model_state_dict' key
        state = ckpt.get('model_state_dict', ckpt)
        self.net.load_state_dict(state)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray):
        """
        Parameters
        ----------
        image_bgr : H×W×3 uint8 BGR image (OpenCV format)

        Returns
        -------
        keypoints  : (N, 2) float32 array of (x, y) pixel coordinates
        descriptors: (N, 256) float32 array
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.TARGET_W, self.TARGET_H))
        inp = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        inp = inp.to(self.device)

        kp_map, desc_map = self.net(inp)  # (1,65,H/8,W/8), (1,256,H/8,W/8)

        H8, W8 = kp_map.shape[2], kp_map.shape[3]

        # Reshape keypoint dustbin channels → (H, W) score map
        kp_np = kp_map[0, :-1, :, :].cpu().numpy()   # (64, H/8, W/8)
        kp_np = kp_np.transpose(1, 2, 0)              # (H/8, W/8, 64)
        kp_np = kp_np.reshape(H8, W8, 8, 8)
        kp_np = kp_np.transpose(0, 2, 1, 3).reshape(H8 * 8, W8 * 8)  # (H, W)

        # NMS
        xs, ys = np.where(kp_np > self.KEYPOINT_THRESHOLD)
        scores = kp_np[xs, ys]
        # Keep top-K
        if len(scores) > self.MAX_KEYPOINTS:
            idx = np.argsort(scores)[-self.MAX_KEYPOINTS:]
            xs, ys, scores = xs[idx], ys[idx], scores[idx]

        keypoints = np.stack([ys, xs], axis=1).astype(np.float32)  # (N, 2) → (x, y)

        if len(keypoints) == 0:
            return keypoints, np.zeros((0, 256), dtype=np.float32)

        # Sample descriptors at keypoint locations
        desc_np = desc_map[0].cpu().numpy()  # (256, H/8, W/8)
        kp_norm = keypoints.copy()
        kp_norm[:, 0] = (kp_norm[:, 0] / (self.TARGET_W - 1)) * 2 - 1
        kp_norm[:, 1] = (kp_norm[:, 1] / (self.TARGET_H - 1)) * 2 - 1
        kp_t = torch.from_numpy(kp_norm).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,N,2)
        sampled = torch.nn.functional.grid_sample(
            desc_map, kp_t, align_corners=True
        )  # (1, 256, 1, N)
        descriptors = sampled[0, :, 0, :].T.cpu().numpy()  # (N, 256)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8

        return keypoints, descriptors


# --------------------------------------------------------------------------- #
# MambaGlue wrapper
# --------------------------------------------------------------------------- #

class MambaGlueMatcher:
    """Thin wrapper around the MambaGlue model from the glue-factory API."""

    CONFIDENCE_THRESHOLD = 0.5
    MIN_MATCHES = 20

    def __init__(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        self.weights_path = weights_path
        self._model = self._load(weights_path, device)

    def _load(self, weights_path: str, device: torch.device):
        try:
            from gluefactory.models.matchers.mambaglue import MambaGlue as _MG
            model = _MG({'weights': weights_path})
            model.to(device).eval()
            return model
        except ImportError as exc:
            raise RuntimeError(
                "MambaGlue (glue-factory) is not installed. "
                "Run: git clone https://github.com/url-kaist/MambaGlue && "
                "cd MambaGlue && pip install -e ."
            ) from exc

    @torch.no_grad()
    def match(
        self,
        kp0: np.ndarray, desc0: np.ndarray,
        kp1: np.ndarray, desc1: np.ndarray,
        image_size: tuple,
    ):
        """
        Parameters
        ----------
        kp0, kp1     : (N, 2) float32 (x, y) keypoints
        desc0, desc1 : (N, 256) float32 descriptors
        image_size   : (W, H) of the images

        Returns
        -------
        pts0, pts1 : matched (M, 2) float32 pixel pairs, or None if < MIN_MATCHES
        """
        def _to_tensor(arr):
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        W, H = image_size
        data = {
            'keypoints0': _to_tensor(kp0),
            'keypoints1': _to_tensor(kp1),
            'descriptors0': _to_tensor(desc0),
            'descriptors1': _to_tensor(desc1),
            'image_size0': torch.tensor([[W, H]], dtype=torch.float32, device=self.device),
            'image_size1': torch.tensor([[W, H]], dtype=torch.float32, device=self.device),
        }
        pred = self._model(data)
        matches = pred['matches0'][0].cpu().numpy()     # (N,)  index into kp1, -1 = unmatched
        scores  = pred['matching_scores0'][0].cpu().numpy()  # (N,) confidence

        valid = (matches > -1) & (scores > self.CONFIDENCE_THRESHOLD)
        if valid.sum() < self.MIN_MATCHES:
            return None, None

        pts0 = kp0[valid]
        pts1 = kp1[matches[valid]]
        return pts0, pts1


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

class _Timer:
    def __init__(self):
        self.elapsed: dict[str, float] = {}

    def measure(self, key: str, fn):
        t0 = time.perf_counter()
        result = fn()
        self.elapsed[key] = (time.perf_counter() - t0) * 1000  # ms
        return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class VOInference:
    """
    Main inference object.  Initialise once; call `estimate_pose` per frame pair.

    Parameters
    ----------
    superpoint_weights : path to superpoint.pth
    mambaglue_weights  : path to mambaglue_checkpoint_best.tar
    camera_matrix      : 3×3 float32 numpy array (K)
    device             : 'cuda' or 'cpu'
    """

    IMAGE_W = 640
    IMAGE_H = 480

    def __init__(
        self,
        superpoint_weights: str,
        mambaglue_weights: str,
        camera_matrix: np.ndarray,
        device: str = 'cuda',
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.K = camera_matrix.astype(np.float64)
        self.superpoint = SuperPoint(superpoint_weights, self.device)
        self.matcher = MambaGlueMatcher(mambaglue_weights, self.device)
        self.timings: dict[str, float] = {}

    def estimate_pose(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
    ) -> np.ndarray | None:
        """
        Compute the 4×4 relative pose T such that p1 ≈ T @ p0.

        Returns None if the frame pair is degenerate (too few matches or
        too few RANSAC inliers).
        """
        timer = _Timer()

        kp0, desc0 = timer.measure('superpoint_ms', lambda: self.superpoint(frame0))
        kp1, desc1 = timer.measure('superpoint_ms',  # overwrite — same stage
                                   lambda: self.superpoint(frame1))
        # Re-measure both SP calls combined
        t_sp = time.perf_counter()
        kp0, desc0 = self.superpoint(frame0)
        kp1, desc1 = self.superpoint(frame1)
        timer.elapsed['superpoint_ms'] = (time.perf_counter() - t_sp) * 1000

        if len(kp0) < 10 or len(kp1) < 10:
            return None

        t_mg = time.perf_counter()
        pts0, pts1 = self.matcher.match(
            kp0, desc0, kp1, desc1,
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        timer.elapsed['mambaglue_ms'] = (time.perf_counter() - t_mg) * 1000

        if pts0 is None:
            return None

        t_geo = time.perf_counter()
        pose = _recover_pose(pts0, pts1, self.K)
        timer.elapsed['geometry_ms'] = (time.perf_counter() - t_geo) * 1000

        self.timings = timer.elapsed
        return pose


def _recover_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_threshold: float = 1.0,
) -> np.ndarray | None:
    """
    Recover 4×4 pose from matched pixel pairs using the Essential Matrix.

    Returns None if fewer than 8 RANSAC inliers remain.
    """
    E, mask = cv2.findEssentialMat(
        pts0, pts1, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    if E is None or mask is None:
        return None

    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8:
        return None

    _, R, t, _ = cv2.recoverPose(E, pts0[inliers], pts1[inliers], K)

    # Assemble 4×4 homogeneous transform
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


# --------------------------------------------------------------------------- #
# Quick smoke-test (run as __main__)
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python inference.py <superpoint.pth> <mambaglue.tar> "
            "<frame0.png> <frame1.png>"
        )
        sys.exit(1)

    sp_w, mg_w, img0_path, img1_path = sys.argv[1:5]

    # Default Gazebo camera intrinsics (640×480, 80° FOV)
    fx = fy = 554.254
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    vo = VOInference(sp_w, mg_w, K, device='cuda')

    f0 = cv2.imread(img0_path)
    f1 = cv2.imread(img1_path)

    T = vo.estimate_pose(f0, f1)
    if T is None:
        print("Degenerate frame pair — no valid pose returned.")
    else:
        print("Relative pose T (4×4):")
        print(T)
        print(f"\nTimings: {vo.timings}")
