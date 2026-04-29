# Real-Time Visual Odometry Using State Space Models

A ROS2-based visual odometry pipeline using **MambaGlue** — an SSM (State Space Model) based feature matcher — combined with classical geometric pose estimation. The system runs inside a Gazebo simulation and is evaluated against ground truth using the `evo` trajectory evaluation tool. [Link to Paper](https://docs.google.com/document/d/1OksWxSOJlJKUw4DwF5B7yOIgpqpqfljDETu-vJPCxZc/edit?tab=t.cf70makguuna)

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
                       │  keypoints, descriptors (frame N and N-1)
                       ▼
          ┌────────────────────────┐
          │  MambaGlue  (SSM)      │  ← Core SSM component
          │  Mamba S6 matcher      │    Replaces attention with selective
          │  ICRA 2025             │    state space model for matching
          └────────────┬───────────┘
                       │  matched keypoint pairs + confidence scores
                       ▼
          ┌────────────────────────┐
          │   Pose Estimator       │
          │  Essential Matrix      │  cv2.findEssentialMat (RANSAC)
          │  + cv2.recoverPose     │  → R, t (relative pose, up to scale)
          └────────────┬───────────┘
                       │  4×4 homogeneous transform
                       ▼
          ┌────────────────────────┐
          │  Trajectory Integrator │  Cumulative pose: T_world = T_world × T_rel
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

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Robot Framework | ROS2 Humble |
| Simulation | Gazebo Fortress |
| SSM Feature Matcher | MambaGlue (ICRA 2025) |
| Keypoint Detector | SuperPoint |
| Pose Recovery | OpenCV (Essential Matrix + RANSAC) |
| Deep Learning | PyTorch 2.x + CUDA |
| SSM Library | `mamba-ssm`, `causal-conv1d` |
| Image Bridge | `cv_bridge` |
| Trajectory Evaluation | `evo` |
| GPU Profiling | `nvidia-ml-py3` |


---

## Implementation Plan (2 Weeks)


### Week 2 — Integrate, Benchmark, Evaluate

---

#### Day 8–9: ROS2 Node Integration

Goal: `vo_node.py` runs live in ROS2, publishing odometry at the camera framerate.

**`vo_node.py` behaviour:**
- Subscribe to `/camera/image_raw`
- Convert each frame with `cv_bridge` → OpenCV
- Maintain a one-frame buffer (previous frame)
- On each new frame: call `inference.py`, accumulate pose via matrix multiplication, publish result to `/vo/odometry` (nav_msgs/Odometry)
- Log per-frame wall-clock latency to `/vo/latency` (std_msgs/Float64)
- If `inference.py` returns `None` (degenerate frame), hold the last valid pose and increment a dropped-frame counter

**`profiler.py`:**
- Background thread using `nvidia-ml-py3`
- Poll GPU utilisation (%) and VRAM usage (MB) every 500ms
- Write timestamped rows to `results/gpu_log.csv`

**Exit criterion:** `ros2 topic echo /vo/odometry` shows changing poses during live robot movement. Node runs for 5 continuous minutes without crashing.

---

#### Day 10–11: Benchmarking

Goal: produce all computational performance numbers.

**Metrics to record:**

| Metric | How to measure |
|---|---|
| End-to-end FPS | `1000 / mean_latency_ms` from `/vo/latency` |
| Inference latency (mean, std, P95) | Logged per frame in `vo_node.py` |
| SuperPoint time | `time.perf_counter` around SuperPoint call in `inference.py` |
| MambaGlue time | `time.perf_counter` around MambaGlue call |
| Geometry time | `time.perf_counter` around Essential Matrix + recoverPose |
| GPU utilisation | Mean and peak from `results/gpu_log.csv` |
| VRAM usage | Peak MB from `results/gpu_log.csv` |
| Dropped frames | Counter in `vo_node.py` (frames with < 20 inliers) |

Run the node on the pre-collected image dataset (via a ROS2 bag replay) for a reproducible benchmark — live teleoperation introduces variance in robot speed.

---

#### Day 12–13: Trajectory Evaluation (ATE)

Goal: quantify odometry accuracy against Gazebo ground truth.

**Steps:**

1. During a benchmark run, `vo_node.py` also writes predicted poses to `results/predicted_trajectory.txt` in TUM format.

2. Run evaluation:
   ```bash
   evo_ape tum data/groundtruth.txt results/predicted_trajectory.txt \
       --align --correct_scale --plot --save_results results/evo_report/
   ```

3. `evaluate_ate.py` wraps the above, prints a summary table, and saves the trajectory overlay plot.

**Flags explained:**
- `--align`: align the two trajectories in SE3 before computing error (removes initial pose offset)
- `--correct_scale`: correct monocular scale ambiguity by finding the best scalar factor
- `--plot`: save trajectory overlay image

**Expected output:**

```
APE w.r.t. translation part (m)
       max    0.XXX m
      mean    0.XXX m
    median    0.XXX m
       min    0.XXX m
      rmse    0.XXX m
       sse    0.XXX m²
       std    0.XXX m
```

---

#### Day 14: Cleanup

**Tasks:**
- Final results table in README
- Architecture diagram
- Setup instructions verified on a clean shell
- Commit all source code (no model weights or dataset files — these are gitignored)

---

## Evaluation Metrics Summary

### Computational

| Metric | Target (RTX 4090) |
|---|---|
| End-to-end FPS | > 25 FPS (real-time threshold) |
| MambaGlue latency | < 10 ms |
| Total inference latency | < 40 ms |
| GPU utilisation | < 30% |
| Peak VRAM | < 2 GB |

### Accuracy

| Metric | Notes |
|---|---|
| ATE RMSE | Lower is better. Scale-corrected. |
| Dropped frame rate | Frames with < 20 RANSAC inliers / total frames |

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| SSM component | MambaGlue | Only Mamba-based VO-adjacent model with released code and weights |
| Keypoint detector | SuperPoint | Native MambaGlue integration, pretrained, GPU-accelerated |
| Pose recovery | Essential Matrix + RANSAC | Classical, robust, requires no training or fine-tuning |
| Scale handling | `evo --correct_scale` | Monocular scale ambiguity is unavoidable; correct at evaluation time |
| Simulation world | Textured indoor environment | Featureless worlds produce zero SuperPoint detections |
| Ground truth format | TUM (`timestamp tx ty tz qx qy qz qw`) | Native `evo` format, no conversion step |
| Inference module | ROS-free standalone | Keeps `inference.py` independently testable and benchmarkable |

---

## Known Limitations

**Monocular scale ambiguity:** Monocular VO cannot recover metric scale from images alone. The ATE evaluation uses `--correct_scale` to find the best-fit scalar before computing error. All reported trajectory errors are scale-corrected. This is standard practice in monocular VO evaluation.

**Gazebo domain gap:** MambaGlue was trained on real-world outdoor image pairs (MegaDepth, HPatches). Gazebo's rendered textures are synthetic and Phong-shaded. Match quality may degrade in textureless regions of the simulation. A textured indoor world mitigates this but does not eliminate it.

**Pure rotation degeneracy:** The Essential Matrix requires non-zero translation between frames. Pure rotation (robot spinning in place) makes the Essential Matrix ill-defined. The node holds the last valid pose in these cases.

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

# 5. Install remaining Python dependencies
pip install -r vo_ros2_ws/requirements.txt

# 6. Install ROS2 bridge packages
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv

# 7. Download pretrained weights into models/
#    - SuperPoint:  models/superpoint.pth
#    - MambaGlue:   models/mambaglue_checkpoint_best.tar

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
ros2 launch data_collector collect.launch.py
# Teleoperate the robot for 2-3 minutes
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Run VO Node (live):**
```bash
python scripts/run_offline.py

# Running live
ros2 launch ssm_vo vo.launch.py

```


**Evaluate ATE:**
```bash
python scripts/evaluate_ate.py \
    --gt data/groundtruth.txt \
    --pred results/predicted_trajectory.txt

# Visualize trajectory
python scripts/visualize_trajectory.py 
```


**Benchmark inference (standalone, no ROS):**
```bash
python scripts/benchmark_inference.py --data_dir data/images --n_pairs 500
```
---

## Results

*To be filled after evaluation runs.*

| Metric | Value |
|---|---|
| Mean inference latency | — ms |
| End-to-end FPS | — |
| SuperPoint time | — ms |
| MambaGlue time | — ms |
| Geometry time | — ms |
| GPU utilisation (mean) | —% |
| Peak VRAM | — MB |
| ATE RMSE (scale-corrected) | — m |
| Dropped frames | —% |

---

## References

1. **MambaGlue:** Kim et al., *"MambaGlue: Fast and Robust Local Feature Matching with Mamba"*, ICRA 2025. [github.com/url-kaist/MambaGlue](https://github.com/url-kaist/MambaGlue)
2. **MambaVO:** Wang et al., *"MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing"*, CVPR 2025. [arXiv:2412.20082](https://arxiv.org/abs/2412.20082)
3. **Mamba:** Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*, 2023. [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
4. **SuperPoint:** DeTone et al., *"SuperPoint: Self-Supervised Interest Point Detection and Description"*, CVPR Workshops 2018.
5. **evo:** Grupp, *"evo: Python package for the evaluation of odometry and SLAM"*. [github.com/MichaelGrupp/evo](https://github.com/MichaelGrupp/evo)
6. **Awesome Learning-based VO/VIO:** [github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO)
