# Soft Arm Obstacle

`soft_arm_obstacle` 是一个连续体软臂避障与目标位姿到达环境，面向模型式控制、强化学习、接触建模和 TDCR/PINN 原型验证。

项目当前包含两条任务线：

- `arm2d`：二维 PCC 软臂，使用基座推进和左右腱长度增量控制，支持圆障碍、目标位置/朝向、启发式接触和优化式接触后端。
- `arm3d`：三维 TDCR 风格软臂，默认 3 段、每段 3 根肌腱，使用 9 维张力动作，支持球/椭球障碍、目标位置/姿态、PCC3D、PCCA contact、PINN、hybrid 和 oracle 适配后端。

任务目标是到达目标位姿，而不是只接近目标点：2D 环境要求末端位置和 `target_angle` 同时满足阈值；3D 环境要求末端位置和 `target_rotvec` 同时满足阈值。

## 快速开始

创建环境：

```bash
conda env create -f environment.yml
conda activate soft_arm
```

运行 2D 随机动作演示：

```bash
python main.py --task arm2d --render-mode human
```

运行无界面 smoke test：

```bash
python main.py --task arm2d --render-mode none --episodes 1 --steps 20 --seed 0
```

运行 2D 固定接触场景：

```bash
python main.py --task arm2d --scenario-name contact_left --render-mode human
```

切换 2D 优化式接触后端：

```bash
python main.py --task arm2d --scenario-name contact_left --contact-backend optimization --render-mode human
```

运行 3D TDCR 演示：

```bash
python main.py --task arm3d --solver-backend pcc3d --scenario-name reach_easy_3d --render-mode human
```

运行 3D contact-aware hybrid 后端：

```bash
python main.py --task arm3d --solver-backend hybrid --scenario-name contact_sphere_3d --render-mode human
```

`main.py` 用环境动作空间采样随机动作，适合检查环境、渲染和接触诊断；训练和控制器入口在 `train.py`。

## 项目结构

```text
soft_arm_obstacle/
├── common/                 # 2D 几何、PCC 链和通用工具
├── docs/                   # 论文和参考资料
├── examples/
│   ├── manual_control_2d.py
│   └── run_apf_baseline.py
├── tasks/
│   ├── base_env.py
│   ├── arm2d/              # 2D PCC 环境、接触、渲染、APF 和 CEM
│   └── arm3d/              # 3D TDCR 环境、渲染和 CEM
├── tdcr/
│   ├── datasets/           # contact-aware 3D 数据格式和生成脚本
│   ├── pinn/               # PINN 网络与损失脚手架
│   └── solvers/            # PCC3D、PCCA、PINN、hybrid、oracle 适配器
├── tests/                  # 单元测试和 CLI smoke test
├── main.py                 # 环境演示入口
├── train.py                # model_based_cem / PPO 训练与回放入口
└── environment.yml
```

## 2D 任务

2D 任务位于 `tasks/arm2d/`。默认软臂是 2 段 PCC，每段长度 `10.0`，基座沿 `+x` 方向推进。动作是：

```text
[base_motor, seg0_left_delta, seg0_right_delta, seg1_left_delta, seg1_right_delta]
```

默认控制映射：

```text
base_delta = max(action[0], 0) * base_delta_scale
tendon_length_delta = action[1:] * tendon_delta_scale
kappa_i = (l_right_i - l_left_i) / (2 * tendon_routing_radius * segment_length_i)
```

默认配置在 `tasks/arm2d/config.py` 和 `tasks/arm2d/model_2d.py` 中，关键参数包括：

- `segment_lengths=(10.0, 10.0)`
- `base_x_limits=(0.0, 10.0)`
- `curvature_limit=0.30`
- `tendon_rest_length=10.0`
- `tendon_routing_radius=0.5`
- `tendon_length_limits=(5.0, 15.0)`
- `base_delta_scale=1.0`
- `tendon_delta_scale=0.1`

2D 预设场景：

- `reach_easy`
- `contact_left`
- `contact_channel`

## 2D 接触后端

`heuristic` 是默认后端，路径是 broad-phase 检测、细采样碰撞检测、接触点重构和 `ContactSummary` 汇总。它速度快，适合大量 rollout。

`optimization` 后端参考 OpenTDCRContactModel 的接触优化思想，将每个控制段离散为多个常曲率子段，通过 SLSQP 优化曲率，同时约束障碍 clearance 和腱长差分。它支持 warm-start、局部障碍筛选、多初值重启和失败回退；如果优化失败或解仍明显穿透，会回退到 `heuristic`。

启用后端诊断时，`info["contact_backend_details"]` 会包含：

- `requested_backend`
- `used_backend`
- `status`
- `reason`
- `contact_state`
- `optimization_message`
- `iterations`
- `objective_value`
- `min_clearance_residual`
- `max_tendon_residual`

## 3D TDCR 任务

3D 任务位于 `tasks/arm3d/` 和 `tdcr/`。默认采用 TDCR 风格约定：

- 坐标系：基座在原点，backbone 默认沿 `+z`
- 结构：默认 3 段，每段 3 根肌腱
- 动作：9 维 `[-1, 1]` 动作，映射到 `tau.shape == (3, 3)` 张力
- 目标：`target=(x, y, z)` 和 `target_rotvec`
- 障碍：球或轴对齐椭球 `Obstacle3D(center, radii)`

3D 预设场景：

- `reach_easy_3d`
- `contact_sphere_3d`
- `channel_3d`

3D solver 后端：

- `pcc3d`：快速 3D PCC 近似，适合 smoke test 和大量 rollout。
- `cosserat_static`：Cosserat 静力学适配入口；当前参考绑定不满足部分配置时会给出诊断并回退。
- `pcca_contact_3d`：接触感知 PCCA 优化后端。
- `pinn`：PINN 检查点包装器；未提供项目原生检查点时回退到快速近似。
- `hybrid`：PINN/free shape 加 PCCA 接触修正。
- `oracle`：`tdcr-modeling` 风格适配入口；当前仓库内主要作为接口和诊断路径。

## 控制器与训练

`train.py` 支持两类控制器：

- `model_based_cem`：默认算法，基于当前模拟器做 CEM/MPC 规划。
- `ppo`：Stable-Baselines3 PPO 对照基线。

训练 2D CEM 控制器：

```bash
python train.py --task arm2d --algorithm model_based_cem --timesteps 20000 --n-envs 1
```

训练 2D CEM 并固定接触场景：

```bash
python train.py --task arm2d --algorithm model_based_cem --scenario-name contact_left --contact-backend optimization --timesteps 20000 --n-envs 1
```

训练 3D CEM 控制器：

```bash
python train.py --task arm3d --algorithm model_based_cem --solver-backend pcc3d --timesteps 20000 --n-envs 1
```

训练 2D PPO：

```bash
python train.py --task arm2d --algorithm ppo --timesteps 20000 --n-envs 4
```

训练 3D PPO：

```bash
python train.py --task arm3d --algorithm ppo --solver-backend pcc3d --timesteps 20000 --n-envs 4
```

回放已保存控制器：

```bash
python train.py --play --model-path models/best_model --play-render-mode human
```

CEM 控制器保存为 `*.mpc.json`，PPO 模型保存为 `*.zip`。`models/run_config.json` 会记录训练时的任务、观测、后端和规划参数，回放时会自动读取。

常用 CEM 参数：

- `--planning-horizon`
- `--num-candidates`
- `--num-elites`
- `--num-iterations`
- `--planning-gamma`
- `--planning-init-std`
- `--planning-min-std`
- `--planning-momentum`
- `--terminal-distance-weight`
- `--terminal-angle-weight`

## APF 与手动调试

运行 2D 人工势场基线：

```bash
python examples/run_apf_baseline.py --scenario-name baseline --render-mode human
```

列出 APF 示例场景：

```bash
python examples/run_apf_baseline.py --list-scenarios
```

显示 tip-space 简化势场和力场：

```bash
python examples/run_apf_baseline.py --scenario-name baseline --show-potential --show-force-field
```

运行 2D 手动控制器：

```bash
python examples/manual_control_2d.py --scenario-name contact_left
```

手动控制快捷键：

- `W/S`：控制第 1 段左右腱差分
- `A/D`：控制第 2 段左右腱差分
- `Left/Right`：控制基座平移
- `R`：重置场景
- `Esc`：退出

## 观测、奖励和评估

2D 观测模式支持：

- `absolute`
- `relative`
- `hybrid`
- `legacy_v1`

`train.py` 暴露的观测相关参数：

- `--observation-mode`
- `--backbone-points`
- `--disable-backbone-points`
- `--disable-segment-contact-features`

观测可以包含末端位姿、目标相对位置、目标朝向、基座位置、曲率或张力、障碍相对特征、接触摘要、骨架 keypoints、分段 clearance/contact mask 和碰撞计数。

奖励主要由位置进展、距离误差、朝向/姿态误差、动作惩罚、时间惩罚、接触推进奖励、空转接触惩罚、穿透/重叠惩罚和成功奖励组成。

`info["evaluation_metrics"]` 会额外返回路径长度、路径效率、最终距离、最终角度/姿态误差、平均 clearance、最小 clearance、接触比例、最大穿透和到达时间等指标。

## 数据集与 PINN 脚手架

3D contact-aware 数据格式位于 `tdcr/datasets/schema.py`，核心字段包括：

```text
tau:            (3, 3)
s:              (num_points,)
p:              (num_points, 3)
R:              (num_points, 3, 3)
u, v:           (num_points, 3)
segment_id:     (num_points,)
contact_slots:  (max_contacts, 24)
obstacle_slots: (max_obstacles, 10)
robot_params:   (10,)
```

生成一份小型 3D oracle/PCCA 数据集：

```bash
python -m tdcr.datasets.generate_oracle --num-samples 64 --backend pcca_contact_3d
```

也可以指定输出路径和后端：

```bash
python -m tdcr.datasets.generate_oracle \
  --output-path data/tdcr3d_oracle_samples.npz \
  --num-samples 64 \
  --backend cosserat_static
```

PINN 网络脚手架位于 `tdcr/pinn/network.py`，损失项位于 `tdcr/pinn/losses.py`。输入面向 `s + tau + contact_slots + obstacle_slots + robot_params`，输出包含位置、姿态 rotvec、应变和接触力近似项。

## Python API 示例

创建 2D 环境：

```python
from tasks.arm2d import (
    ArmModel2DConfig,
    ContactResponse2DConfig,
    Observation2DConfig,
    Render2DConfig,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
)

env = SoftArmObstacle2DEnv(
    render_mode="rgb_array",
    model_config=ArmModel2DConfig(
        contact_response=ContactResponse2DConfig(backend="optimization"),
    ),
    env_config=SoftArmObstacle2DEnvConfig(
        observation=Observation2DConfig(mode="hybrid", backbone_points=4),
        render=Render2DConfig(xlim=(-5.0, 36.0), ylim=(-16.0, 16.0)),
    ),
)

obs, info = env.reset(seed=0, options={"scenario_name": "contact_left"})
```

创建 3D 环境：

```python
from tasks.arm3d import ArmModel3DConfig, SoftArmObstacle3DEnv

env = SoftArmObstacle3DEnv(
    render_mode="rgb_array",
    max_steps=200,
    model_config=ArmModel3DConfig(solver_backend="pcc3d"),
)

obs, info = env.reset(seed=0, options={"scenario_name": "reach_easy_3d"})
```

## 测试

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 2D 正向运动学和腱长到曲率映射
- 2D/3D 环境 reset、step 和目标成功判据
- 接触响应数值鲁棒性
- 优化式接触后端的腱长约束和 clearance
- 固定场景复现
- `main.py` / `train.py` CLI smoke test

## 输出与缓存

常见生成目录：

- `models/`：训练得到的 PPO 和 CEM 控制器
- `logs/`：TensorBoard 与评估日志
- `data/`：生成的 TDCR/PINN 数据集

建议 `.gitignore` 忽略 Python 缓存和运行产物：

```gitignore
__pycache__/
*.py[cod]
models/
logs/
data/
```
