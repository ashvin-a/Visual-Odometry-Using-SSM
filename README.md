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

## Repository Structure

```
vo_ros2_ws/
├── src/
│   ├── ssm_vo/                        # Core ROS2 package
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── resource/ssm_vo
│   │   ├── ssm_vo/
│   │   │   ├── __init__.py
│   │   │   ├── vo_node.py             # ROS2 node: image_raw → /vo/odometry
│   │   │   ├── inference.py           # Standalone inference (no ROS dependency)
│   │   │   ├── pose_estimator.py      # Essential matrix, recoverPose, accumulation
│   │   │   └── profiler.py            # FPS, latency, GPU utilization logging
│   │   └── launch/
│   │       └── vo.launch.py           # Launches VO node
│   │
│   ├── robot_description/             # Robot + world definition
│   │   ├── urdf/
│   │   │   └── diffbot_camera.urdf.xacro
│   │   ├── worlds/
│   │   │   └── structured_env.world   # Textured indoor environment
│   │   └── launch/
│   │       └── spawn_robot.launch.py
│   │
│   └── data_collector/                # Dataset recording package
│       ├── package.xml
│       ├── setup.py
│       ├── launch/
│       │   └── collect.launch.py      # Launches image_saver + gt_pose_saver
│       └── data_collector/
│           ├── image_saver_node.py    # /camera/image_raw → disk
│           └── gt_pose_saver_node.py  # /odom → TUM format file
│
├── scripts/
│   ├── evaluate_ate.py                # Wraps evo_ape, prints ATE table
│   ├── benchmark_inference.py         # Standalone latency/FPS benchmark
│   └── visualize_trajectory.py        # Plots predicted vs ground truth path
│
├── models/                            # Pretrained weights — not committed to git
│   ├── superpoint.pth
│   └── mambaglue_checkpoint_best.tar
│
├── data/                              # Collected dataset — not committed to git
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   └── groundtruth.txt               # TUM format: timestamp tx ty tz qx qy qz qw
│
├── results/                           # Benchmark outputs
│   ├── latency_log.csv
│   ├── gpu_log.csv
│   ├── predicted_trajectory.txt
│   └── evo_report/
│
├── requirements.txt
└── README.md
```

---

## Implementation Plan (2 Weeks)

### Week 1 — Build the Pipeline

---

#### Day 1–2: Environment Setup

Goal: all dependencies install and run before writing a single line of project code.

**Steps:**

1. Verify ROS2 Humble is installed and sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 doctor
   ```

2. Verify GPU is accessible:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

3. Install PyTorch with CUDA 12.4 support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. Install Mamba SSM libraries:
   ```bash
   pip install mamba-ssm causal-conv1d --no-build-isolation
   ```

5. Clone and install MambaGlue:
   ```bash
   git clone https://github.com/url-kaist/MambaGlue
   cd MambaGlue && pip install -e .
   # Download checkpoint_best.tar from the glue-factory branch if not in main
   ```

6. Run MambaGlue on two test images to confirm it produces match pairs.

7. Install remaining dependencies:
   ```bash
   pip install evo nvidia-ml-py3 opencv-python kornia
   sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv
   ```

**Exit criterion:** MambaGlue runs on two test images and returns match pairs. Gazebo opens with no errors.

> **Risk:** MambaGlue pretrained weights may be in the `glue-factory` training branch rather than the main branch. Check this on Day 1 — not on Day 8.

---

#### Day 3–4: Simulation Setup

Goal: a robot drives around Gazebo and publishes `/camera/image_raw` and `/odom`.

**Steps:**

1. Write `diffbot_camera.urdf.xacro`:
   - Differential-drive robot base
   - `camera_link` with `libgazebo_ros_camera` plugin publishing `/camera/image_raw` at 30Hz (640×480, RGB8)
   - `libgazebo_ros_diff_drive` publishing `/odom` (ground truth odometry)

2. Select or create a textured Gazebo world. **Do not use a blank world** — blank Gazebo environments have featureless walls with no texture gradient. SuperPoint will detect zero keypoints and the pipeline will produce no matches.
   - Minimum requirement: varied surface textures, furniture, edges

3. Write `spawn_robot.launch.py` to launch Ignition Gazebo Fortress with the world, spawn the robot, and bridge all required topics.

4. Verify:
   ```bash
   ros2 topic hz /camera/image_raw    # should show ~30Hz
   ros2 run rqt_image_view rqt_image_view
   ros2 topic echo /odom              # should change when robot is driven
   ```

**Exit criterion:** `teleop_twist_keyboard` drives the robot and the camera image appears in `rqt_image_view` showing the simulated environment.

---

#### Day 5: Data Collection

Goal: collect a ground truth dataset for offline ATE evaluation.

**Steps:**

1. `image_saver_node.py`: subscribe to `/camera/image_raw`, save frames as `{timestamp:.6f}.png` to `data/images/`

2. `gt_pose_saver_node.py`: subscribe to `/odom`, write each pose in TUM format:
   ```
   timestamp tx ty tz qx qy qz qw
   ```
   to `data/groundtruth.txt`

3. Drive a 2–3 minute trajectory through the environment. Cover varied directions — straight lines, curves, and one full loop. Avoid pure rotation-in-place (degenerate for monocular VO).

4. Verify dataset: approximately 3600+ images, a matching ground truth file with one line per `/odom` message.

> **Note on scale:** Monocular VO cannot recover metric scale from images alone. The predicted trajectory will be correct in shape but wrong in absolute units. This is handled at evaluation time using `evo`'s `--correct_scale` flag. It is expected behaviour, not a failure.

---

#### Day 6–7: Core Inference Pipeline

Goal: `inference.py` takes two images and returns a 4×4 relative pose matrix. Fully standalone — no ROS dependency.

**Steps:**

1. **SuperPoint stage:** load pretrained SuperPoint, extract keypoints and 256-dim descriptors from each input frame. Resize frames to 640×480 and normalise to [0, 1] greyscale before passing to the detector.

2. **MambaGlue stage:** pass both sets of keypoints and descriptors through MambaGlue. Filter output matches by confidence score (threshold: 0.5). Require a minimum of 20 matches; if below this, return `None` (degenerate frame).

3. **Pose estimator (`pose_estimator.py`):**
   - `cv2.findEssentialMat` with RANSAC to compute the Essential Matrix from matched point pairs and camera intrinsics
   - `cv2.recoverPose` to decompose into R and t (translation up to scale)
   - Assemble into a 4×4 homogeneous transform

4. **Camera intrinsics:** read from the Gazebo camera plugin parameters (fx, fy, cx, cy). Pass as a config dict rather than hardcoding.

5. **`benchmark_inference.py`:** run 500 consecutive frame pairs from the collected dataset, record wall-clock time per pair, report mean, std, P95 latency, and derived FPS.

**Exit criterion:** `inference.py` on two consecutive frames from the dataset prints a plausible R (close to identity for slow motion) and a unit-length t. Benchmark script runs 500 pairs without error.

---

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
cd vo_ros2_ws

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Mamba SSM libraries
pip install mamba-ssm causal-conv1d --no-build-isolation

# 4. Install MambaGlue
git clone https://github.com/url-kaist/MambaGlue
cd MambaGlue && pip install -e . && cd ..

# 5. Download pretrained weights into models/
#    - SuperPoint:  models/superpoint.pth
#    - MambaGlue:   models/mambaglue_checkpoint_best.tar

# 6. Build the ROS2 workspace
source /opt/ros/humble/setup.bash
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

**Run VO node (live):**
```bash
ros2 launch ssm_vo vo.launch.py
```

**Benchmark inference (standalone, no ROS):**
```bash
python scripts/benchmark_inference.py --data_dir data/images --n_pairs 500
```

**Evaluate ATE:**
```bash
python scripts/evaluate_ate.py \
    --gt data/groundtruth.txt \
    --pred results/predicted_trajectory.txt
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
