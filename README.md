# Real-Time Visual Odometry Using State Space Models

A ROS2-based visual odometry pipeline using **MambaGlue** — an SSM (State Space Model) based feature matcher — combined with classical geometric pose estimation. The system runs inside a Gazebo simulation and is evaluated against ground truth using the `evo` trajectory evaluation tool. [Link to Paper](https://github.com/ashvin-a/Visual-Odometry-Using-SSM/blob/main/Real_time_Visual_Odometry_using_State_Space_Models.pdf)

---

## Research Question

> How do State Space Model-based feature matchers (MambaGlue) perform in a real-time visual odometry pipeline compared to attention-based alternatives, within a ROS2/Gazebo environment?

### Why MambaGlue, not MambaVO?

MambaVO (CVPR 2025) is the natural target for this project — it is a complete, end-to-end Mamba-based visual odometry system. However, as of the time of writing, no code or pretrained weights have been released by the authors. Every other end-to-end Mamba VO paper is in the same state.

**MambaGlue** (ICRA 2025) is the only Mamba-based model in the visual odometry pipeline space with released code and pretrained weights. It replaces the attention mechanism in SuperGlue with a selective SSM (S6) layer for keypoint correspondence — exactly the stage where the "is SSM fast enough for real-time?" question is most interesting to answer. This makes the research question sharper, not weaker:

- SSM-based matching has **O(N) complexity** vs. O(N²) for attention
- The matching stage is a real bottleneck in feature-based VO
- A direct latency comparison between MambaGlue and SuperGlue/LightGlue is a measurable, falsifiable result

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gazebo Simulation                        │
│   Differential-drive robot + monocular RGB camera (30Hz)        │
└───────────────────────┬─────────────────────────────────────────┘
                        │  /camera/image_raw  (sensor_msgs/Image)
                        ▼
              ┌─────────────────┐
              │    cv_bridge    │  ROS2 → OpenCV conversion
              └────────┬────────┘
                       │  numpy array (H × W × 3)
                       ▼
              ┌─────────────────┐
              │   SuperPoint    │  Keypoint detection + 256-dim descriptors
              └────────┬────────┘
                       │  keypoints, scores, descriptors (frame N and N-1)
                       ▼
          ┌────────────────────────────────────────────────┐
          │  Feature Matcher  (pluggable — select via CLI) │
          │                                                │
          │  MambaGlue  (SSM, ICRA 2025)   ← default      │
          │    Mamba S6 layers, O(N) complexity            │
          │                                                │
          │  SuperGlue  (attention, NeurIPS 2020)          │
          │    GNN + self/cross-attention, O(N²)           │
          │                                                │
          │  LightGlue  (attention, ICCV 2023)             │
          │    Adaptive depth/width attention, O(N²)       │
          └────────────────────┬───────────────────────────┘
                               │  matched keypoint pairs + confidence scores
                       ▼
          ┌────────────────────────┐
          │   Pose Estimator       │
          │  Essential Matrix      │  cv2.findEssentialMat (RANSAC)
          │  + cv2.recoverPose     │  → R, t (relative pose, up to scale)
          │  + rotation gate       │  rejects |rot| > 45° as degenerate
          │  + inlier ratio check  │  rejects < 15% inlier ratio
          └────────────┬───────────┘
                       │  4×4 homogeneous transform
                       ▼
          ┌────────────────────────┐
          │  Trajectory Integrator │  T_world = T_world × T_rel_robot
          │  + planarity clamp     │  zeros roll/pitch/Z each step
          └────────────┬───────────┘
                       │
                       ▼
              /vo/odometry  (nav_msgs/Odometry)
              /vo/latency   (std_msgs/Float64)
                       │
                       ▼
          ┌────────────────────────┐
          │   evo ATE Evaluation   │  vs. /odom Gazebo ground truth
          └────────────────────────┘
```

---

## Installation

```bash
# 1. Clone this repository
git clone <repo-url>
cd Visual-Odometry-Using-SSM

# 2. Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install Mamba SSM libraries (requires torch to be installed first)
pip install mamba-ssm causal-conv1d --no-build-isolation

# 4. Install MambaGlue
git clone https://github.com/url-kaist/MambaGlue mamba_glue
cd mamba_glue && pip install -e . && cd ..

# 5. Install remaining Python dependencies (includes LightGlue)
pip install -r vo_ros2_ws/requirements.txt

# 6. Install ROS2 bridge packages
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv

# 7. Download pretrained weights into models/
#    - SuperPoint:  models/superpoint.pth
#    - MambaGlue:   models/checkpoint_best.tar

# 8. Build the ROS2 workspace
source /opt/ros/humble/setup.bash
cd vo_ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Usage

**Launch simulation:**
```bash
ros2 launch robot_description spawn_robot.launch.py
```

**Collect dataset:**
```bash
# Option A — single launch file (recommended)
ros2 launch data_collector collect.launch.py

# Option B — start each node manually
ros2 run data_collector image_saver       # Terminal 1: save images
ros2 run data_collector gt_pose_saver     # Terminal 2: save ground truth poses

# Terminal 3 (either option): drive the robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Run offline VO on collected images:**
```bash
# MambaGlue — pure monocular (scale-corrected ATE evaluation)
python vo_ros2_ws/scripts/run_offline.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --mg_weights models/checkpoint_best.tar \
    --matcher mambaglue \
    --output results/traj_mambaglue.txt

# SuperGlue baseline
python vo_ros2_ws/scripts/run_offline.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --matcher superglue --sg_weights outdoor --sg_repo superglue \
    --output results/traj_superglue.txt

# LightGlue baseline (full-depth, comparable to SuperGlue)
python vo_ros2_ws/scripts/run_offline.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --matcher lightglue \
    --output results/traj_lightglue.txt

# LightGlue with adaptive pruning (faster, real-time mode)
python vo_ros2_ws/scripts/run_offline.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --matcher lightglue --lg_adaptive \
    --output results/traj_lightglue_adaptive.txt

# GT-scale-assisted mode — add --gt_file to any of the above
python vo_ros2_ws/scripts/run_offline.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --mg_weights models/mambaglue_checkpoint_best.tar \
    --gt_file vo_ros2_ws/data/groundtruth.txt \
    --output results/predicted_trajectory_gt.txt
```

**Run VO node live (ROS2):**
```bash
ros2 launch ssm_vo vo.launch.py
```

**Evaluate ATE:**
```bash
python vo_ros2_ws/scripts/evaluate_ate.py \
    --gt vo_ros2_ws/data/groundtruth.txt \
    --pred results/predicted_trajectory.txt # or results/predicted_trajectory_gt.txt
```

**Visualize trajectory:**
```bash
python vo_ros2_ws/scripts/visualize_trajectory.py \
    --gt vo_ros2_ws/data/groundtruth.txt \
    --pred results/predicted_trajectory.txt \
    --out results/trajectory_plot.png \
    --correct_scale
```

**Benchmark inference (standalone, no ROS):**
```bash
# Run from the repo root — script defaults expect models/ at the root level
python vo_ros2_ws/scripts/benchmark_inference.py \
    --data_dir vo_ros2_ws/data/images \
    --sp_weights models/superpoint.pth \
    --mg_weights models/checkpoint_best.tar \
    --n_pairs 500 \
    --device cuda
```

---

## Results

*To be filled after evaluation runs.*

### Accuracy (Gazebo, scale-corrected ATE)

| Matcher | ATE RMSE (m) | Dropped frames (%) |
|---|---|---|
| MambaGlue (SSM, O(N)) | — | — |
| SuperGlue (attention, O(N²)) | — | — |
| LightGlue full-depth (attention, O(N²)) | — | — |
| LightGlue adaptive | — | — |

### Latency (NVIDIA GPU, per frame pair)

| Matcher | SuperPoint (ms) | Matcher (ms) | Geometry (ms) | Total (ms) | FPS |
|---|---|---|---|---|---|
| MambaGlue | — | — | — | — | — |
| SuperGlue | — | — | — | — | — |
| LightGlue full-depth | — | — | — | — | — |
| LightGlue adaptive | — | — | — | — | — |

### GPU utilisation (MambaGlue run)

| Metric | Value |
|---|---|
| GPU utilisation (mean) | —% |
| Peak VRAM | — MB |

---

## Known Limitations

**Monocular scale ambiguity:** `cv2.recoverPose` always returns a unit-norm translation vector — metric scale cannot be recovered from images alone. Two evaluation modes are supported:

- *Pure monocular:* ATE evaluation uses `--correct_scale` (Umeyama SE3 + scale alignment). All reported errors are scale-corrected; this is standard practice in monocular VO.
- *GT-scale-assisted:* Pass `--gt_file` to `run_offline.py` to scale each relative translation by the ground-truth inter-frame displacement. This isolates rotation accuracy from the scale problem and is clearly labelled in experiments.

**Gazebo domain gap:** MambaGlue was trained on real-world outdoor image pairs (MegaDepth, HPatches). Gazebo's rendered textures are synthetic and Phong-shaded. Match quality may degrade in textureless regions of the simulation. A textured indoor world mitigates this but does not eliminate it.

**Pure rotation degeneracy:** The Essential Matrix requires non-zero translation between frames. Pure rotation (robot spinning in place) makes the Essential Matrix ill-defined. The pipeline drops these frames and holds the last valid pose.

**Degenerate Essential Matrix solutions:** On low-texture or near-planar scenes, `cv2.recoverPose` can return numerically valid but physically impossible solutions (e.g., near-180° rotations). Two guards are in place: an inlier ratio threshold (≥ 15% of matched points must be RANSAC inliers) and a rotation magnitude gate (< 45° per frame); frames that fail either check are dropped.

---

## References

1. **MambaGlue:** Kim et al., *"MambaGlue: Fast and Robust Local Feature Matching with Mamba"*, ICRA 2025. [github.com/url-kaist/MambaGlue](https://github.com/url-kaist/MambaGlue)
2. **SuperGlue:** Sarlin et al., *"SuperGlue: Learning Feature Matching with Graph Neural Networks"*, CVPR 2020. [github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
3. **LightGlue:** Lindenberger et al., *"LightGlue: Local Feature Matching at Light Speed"*, ICCV 2023. [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue)
4. **MambaVO:** Wang et al., *"MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing"*, CVPR 2025. [arXiv:2412.20082](https://arxiv.org/abs/2412.20082)
5. **Mamba:** Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*, 2023. [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
6. **SuperPoint:** DeTone et al., *"SuperPoint: Self-Supervised Interest Point Detection and Description"*, CVPR Workshops 2018.
7. **evo:** Grupp, *"evo: Python package for the evaluation of odometry and SLAM"*. [github.com/MichaelGrupp/evo](https://github.com/MichaelGrupp/evo)
8. **Awesome Learning-based VO/VIO:** [github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO)
