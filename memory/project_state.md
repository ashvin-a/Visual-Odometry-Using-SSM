---
name: Project state — Visual Odometry Using SSM
description: Current implementation state and key architectural decisions for the ROS2 MambaGlue VO project
type: project
---

The full ROS2 workspace `vo_ros2_ws/` has been scaffolded and all source files written. The project implements visual odometry using MambaGlue (ICRA 2025, SSM-based feature matcher) instead of MambaVO (CVPR 2025) because MambaVO has no released code or weights.

**Why:** MambaVO is end-to-end Mamba VO but code is not public. MambaGlue replaces transformer attention with selective SSM (S6) in the matching stage — the bottleneck in feature-based VO — giving a measurable O(N) vs O(N²) comparison point.

**How to apply:** When discussing the project, frame the SSM contribution as MambaGlue's matching stage, not MambaVO. The research question is about SSM speed/accuracy in the matching stage, not end-to-end pose prediction.

## Packages built
- `src/ssm_vo/` — Core ROS2 Python package: `vo_node.py`, `inference.py`, `pose_estimator.py`, `profiler.py`
- `src/robot_description/` — CMake package: URDF xacro, Gazebo world, spawn launch file
- `src/data_collector/` — Python package: `image_saver_node.py`, `gt_pose_saver_node.py`
- `scripts/` — Standalone: `benchmark_inference.py`, `evaluate_ate.py`, `visualize_trajectory.py`

## Next steps (Day 1–2 per README plan)
1. Install dependencies: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
2. Install mamba-ssm: `pip install mamba-ssm causal-conv1d --no-build-isolation`
3. Clone + install MambaGlue: `git clone https://github.com/url-kaist/MambaGlue && cd MambaGlue && pip install -e .`
4. Download weights: SuperPoint (`superpoint.pth`) and MambaGlue checkpoint into `models/`
5. Build workspace: `colcon build --symlink-install`
6. Launch Gazebo: `ros2 launch robot_description spawn_robot.launch.py`

## Camera intrinsics (Gazebo defaults)
640×480, 80° FOV → fx=fy=554.254, cx=320, cy=240
