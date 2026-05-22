# Soft Arm Obstacle

一个连续体软臂障碍环境。2D 任务保留原有 PCC/腱长增量基线；3D 任务已经接入 TDCR 风格的 3 段 9 肌腱张力驱动、`+z` backbone 坐标系、接触感知 PCCA 近似、PINN/tdcr-modeling 适配层和 Gymnasium 强化学习接口。

当前版本已经升级为目标位姿任务：策略不仅要把末端送到目标位置附近，还要把末端朝向对准目标朝向。

## 当前能力

- 2D 分段常曲率软臂建模，默认 2 段，每段长度 `10.0`
- 基座沿 `+x` 方向单向推进
- 左右腱长度驱动，每步动作先更新腱长，再由常曲率关系解出曲率
- 圆障碍随机采样与固定预设场景
- 目标位姿采样：目标位置 `target` + 目标朝向 `target_angle`
- `heuristic` 几何接触后端
- 参考 OpenTDCRContactModel 的 `optimization` 优化式接触后端
- `gymnasium.Env` 接口、模型式控制脚本和 PPO 对照基线
- `absolute` / `relative` / `hybrid` / `legacy_v1` 四种观测布局
- 骨架 keypoints、接触摘要、分段接触特征、碰撞计数
- `human` / `rgb_array` 渲染
- 3D TDCR 任务：默认 3 段，每段 3 根肌腱，动作是 9 维张力输入
- 3D solver 后端：`pcc3d`、`cosserat_static`、`pcca_contact_3d`、`pinn`、`hybrid`、`oracle`
- contact-aware PINN 数据格式：`tau`、障碍槽、接触槽、`p/R/u/v` backbone 标签
- 自动化测试与 CLI smoke test

## 快速开始

```bash
conda env create -f environment.yml
conda activate soft_arm

python main.py --render-mode human
```

无界面 smoke test：

```bash
python main.py --render-mode none --episodes 1 --steps 20 --seed 0
```

运行固定接触场景：

```bash
python main.py --render-mode human --scenario-name contact_left
```

启用优化式接触后端：

```bash
python main.py --render-mode human --scenario-name contact_left --contact-backend optimization
```

运行 3D TDCR 随机动作演示：

```bash
python main.py --task arm3d --solver-backend pcc3d --render-mode human --scenario-name reach_easy_3d
```

运行 3D contact-aware 后端：

```bash
python main.py --task arm3d --solver-backend hybrid --render-mode human --scenario-name contact_sphere_3d
```

默认训练模型式控制器（`model_based_cem`）：

```bash
python train.py --timesteps 20000 --n-envs 4
```

训练 3D 模型式控制器：

```bash
python train.py --task arm3d --solver-backend pcc3d --timesteps 20000 --n-envs 1
```

训练 3D PPO：

```bash
python train.py --task arm3d --algorithm ppo --solver-backend pcc3d --timesteps 20000 --n-envs 4
```

训练时固定场景并启用优化后端：

```bash
python train.py --timesteps 20000 --n-envs 1 --scenario-name contact_left --contact-backend optimization
```

如需运行旧的无模型 PPO 对照基线：

```bash
python train.py --algorithm ppo --timesteps 20000 --n-envs 4
```

回放已训练模型：

```bash
python train.py --play --model-path models/best_model --play-render-mode human
```

`main.py` 默认使用随机动作做环境演示，不是训练好的策略；真正的模型式控制入口在 `train.py` 的默认 `model_based_cem` 控制器。

## 目录结构

```text
soft_arm_obstacle/
├── common/
│   ├── geometry.py
│   ├── pcc.py
│   └── utils.py
├── examples/
│   └── manual_control_2d.py
├── tasks/
│   ├── base_env.py
│   ├── arm2d/
│   │   ├── config.py
│   │   ├── contact_solver_2d.py
│   │   ├── env_2d.py
│   │   ├── model_2d.py
│   │   ├── physics_2d.py
│   │   ├── registration.py
│   │   ├── renderer_2d.py
│   │   └── sim_2d.py
│   └── arm3d/
├── tdcr/
│   ├── actuation.py
│   ├── robot_params.py
│   ├── shapes.py
│   ├── datasets/
│   ├── pinn/
│   └── solvers/
├── tests/
│   ├── test_arm2d_env.py
│   ├── test_arm2d_model.py
│   └── test_arm3d_env.py
├── environment.yml
├── main.py
└── train.py
```

## 核心设计

### 1. 软臂与驱动

- 几何模型：2D PCC, piecewise constant curvature
- 默认参数：
  - `segment_lengths=(10.0, 10.0)`
  - `base_x_limits=(0.0, 10.0)`
  - `curvature_limit=0.18`
  - `arm_radius=0.2`
  - `tendon_rest_length=10.0`
  - `tendon_routing_radius=0.5`
  - `tendon_length_limits=(5.0, 15.0)`

模型的主状态不是直接存曲率，而是每段左右两根腱的当前长度：

```text
free_tendon_lengths[i] = [l_left_i, l_right_i]
```

每段曲率由常曲率近似关系得到：

```text
kappa_i = (l_right_i - l_left_i) / (2 * tendon_routing_radius * segment_length_i)
```

### 2. 动作定义

当前默认是 5 维动作：

```text
[base_motor, seg0_left_delta, seg0_right_delta, seg1_left_delta, seg1_right_delta]
```

默认控制配置在 `tasks/arm2d/config.py`：

- `base_delta_scale = 1.0`
- `tendon_delta_scale = 0.1`
- 动作范围 `[-1, 1]`

动作语义：

- `base_motor` 只允许前进，实际映射是
  `base_delta = max(action[0], 0) * base_delta_scale`
- 其余四维是左右腱的单步长度增量
  `delta_l = action * tendon_delta_scale`

也就是说：

- `action = 1` 表示该腱本步伸长 `0.1`
- `action = -1` 表示该腱本步缩短 `0.1`

### 2.1 3D TDCR 融合路线

3D 任务位于 `tasks/arm3d/`，采用和 `tdcr-modeling` / `tdcr-pinn` 更一致的约定：

- 坐标系：基座在原点，默认 backbone 沿 `+z`
- 结构：默认 `segment_lengths=(0.08, 0.08, 0.08)`，每段 3 根肌腱
- 动作：9 维 `[-1, 1]` 动作映射到张力 `tau.shape == (3, 3)`
- 目标：3D 位置 `target=(x, y, z)` 和末端 `target_rotvec`
- 障碍：球 / 轴对齐椭球 `Obstacle3D(center, radii)`

3D solver 后端：

- `pcc3d`：快速 3D PCC 近似，适合大量 rollout 和 smoke test
- `cosserat_static`：优先接入 `tdcr-modeling` / `pytdcrsv` 的 Cosserat 静力学后端；当前参考绑定主要支持 2 段 6 腱，不满足时会给出诊断并回退到 `pcc3d`
- `pcca_contact_3d`：参考 OpenTDCRContactModel 的接触优化思想，用 SLSQP 优化每段 3D 曲率并满足 clearance 约束
- `pinn`：PINN 检查点包装器；未加载项目原生检查点时自动回退到 `pcc3d`
- `hybrid`：PINN/free shape 加接触 PCCA 修正
- `oracle`：tdcr-modeling 适配入口；当前仓库内参考实现主要是 2 段/6 腱，3 段/9 腱严格 oracle 尚需扩展，默认给出诊断并回退到 `pcc3d`

3D 数据格式在 `tdcr/datasets/schema.py`：

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

可生成一份小型 oracle/PCCA 数据集：

```bash
python -m tdcr.datasets.generate_oracle --num-samples 64 --backend pcca_contact_3d
python -m tdcr.datasets.generate_oracle --num-samples 64 --backend cosserat_static
```

PINN 网络脚手架位于 `tdcr/pinn/network.py`，输入为 `s + tau + contact_slots + obstacle_slots + robot_params`，输出默认包含 `p`、姿态 rotvec、`u`、`v` 和接触力近似项。

### 3. 任务定义：目标位姿

场景由 `Scenario2D` 或 `SoftArmSim2DConfig` 定义。当前任务不是单纯“进入目标点”，而是“到达目标位姿”：

- 目标位置：`target = (x, y)`
- 目标朝向：`target_angle`
- 成功位置容差：`target_radius`
- 成功朝向容差：`target_angle_tolerance`

默认随机采样配置：

- `obstacle_count = 6`
- `target_radius = 0.25`
- `target_angle_tolerance = 0.25` rad
- `target_angle_range = [-pi, pi]`

成功条件：

```text
distance_to_target <= target_radius
and
abs(angle_error_to_target) <= target_angle_tolerance
```

### 4. 接触后端

#### `heuristic`

保留原来的快速几何接触重构路线：

- 先做 broad-phase + 细采样碰撞检测
- 找到首个碰撞点
- 对碰撞段进行切分和贴障几何重构
- 用 `ContactSummary` 输出最小 clearance、接触状态、最近障碍、分段 mask 等摘要

优点是快，适合大量 rollout；缺点是接触物理一致性有限。

#### `optimization`

这条路线参考 OpenTDCRContactModel，但为了接进当前环境，做了 2D/Python 化实现：

- 把每个控制段离散成多个常曲率子段
- 优化变量是所有离散子段的曲率
- 目标函数由三部分组成：
  - 弯曲能量
  - 对自由形状的跟踪项
  - 子段曲率平滑项
- 障碍约束是显式不等式约束：采样骨干和横向包络点，要求都在障碍外侧
- 拉线约束不再是“保平均曲率”，而是更接近 OpenTDCR 的腱长约束：
  - 每个控制段的左右腱平均长度会写入该段离散子段的中性腱长
  - 再用每段左右腱差分作为等式约束，保证接触求解后整段腱长与目标腱长一致
- 支持 warm-start：上一帧优化解会作为下一帧初值
- 支持局部障碍筛选：只把靠近自由形状的障碍送进优化器
- 支持多初值重启、更多约束采样点和基于 clearance 的软惩罚，以提高数值鲁棒性
- 若优化失败或解仍存在明显穿透，可自动回退到 `heuristic`

这一版已经比“保平均曲率”的过渡版本更接近 OpenTDCR 风格。

### 5. 观测、奖励和评估

观测可以包含：

- 末端位置 / 朝向
- 目标相对位置
- 目标朝向向量
- `base_x`
- 各段曲率
- 障碍相对特征
- 接触摘要
- 骨架 keypoints
- 分段 clearance / contact mask
- 碰撞计数

奖励由这些部分组成：

- 位置进展奖励
- 当前位置误差惩罚
- 朝向进展奖励
- 当前朝向误差惩罚
- 动作惩罚
- 时间惩罚
- 接触推进奖励
- 空转接触惩罚
- 重叠惩罚
- 成功奖励

`info["evaluation_metrics"]` 会额外返回：

- `distance_improvement`
- `angle_improvement`
- `final_distance`
- `final_angle_error`
- `path_length`
- `path_efficiency`
- `mean_clearance`
- `min_clearance`
- `contact_ratio`
- `max_penetration`
- `time_to_goal`

### 6. 可视化

`tasks/arm2d/renderer_2d.py` 使用 Matplotlib 绘制：

- 红色圆盘：障碍物
- 绿色圆盘：目标位置容差区域
- 绿色箭头：目标朝向
- 灰色虚线：不考虑接触的自由形状
- 蓝色实线：接触后实际观测形状
- 青绿色 `x`：骨架 keypoints
- 橙色点：碰撞 / 接触点
- 黑色方块：基座
- 蓝色圆点：末端

左上角还会显示当前接触状态、最小 clearance 和主要 penetration。

## 配置入口

主要配置对象：

- `ArmModel2DConfig`
- `ContactResponse2DConfig`
- `SoftArmSim2DConfig`
- `SoftArmObstacle2DEnvConfig`
- `TendonControl2DConfig`
- `Reward2DConfig`
- `Observation2DConfig`
- `Render2DConfig`

示例：

```python
from tasks.arm2d import (
    ArmModel2DConfig,
    ContactResponse2DConfig,
    Observation2DConfig,
    Render2DConfig,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
    TendonControl2DConfig,
)

env = SoftArmObstacle2DEnv(
    render_mode="rgb_array",
    model_config=ArmModel2DConfig(
        contact_response=ContactResponse2DConfig(backend="optimization"),
    ),
    env_config=SoftArmObstacle2DEnvConfig(
        control=TendonControl2DConfig(base_delta_scale=1.0, tendon_delta_scale=0.1),
        observation=Observation2DConfig(mode="hybrid", backbone_points=5),
        render=Render2DConfig(xlim=(-5.0, 30.0), ylim=(-12.0, 12.0)),
    ),
)
```

固定场景重置：

```python
obs, info = env.reset(
    seed=0,
    options={"scenario_name": "contact_left"},
)
```

## 可解释日志与调试输出

当使用 `optimization` 后端时，`info["contact_backend_details"]` 会返回结构化诊断信息，包括：

- `requested_backend`
- `used_backend`
- `status`
- `reason`
- `contact_state`
- `constraint_mode`
- `optimization_message`
- `reduced_obstacle_count`
- `iterations`
- `objective_value`
- `max_tendon_residual`
- `max_tendon_differential_residual`
- `min_clearance_residual`

如果发生回退，`reason` 会明确说明是：

- 优化器失败
- 解不满足可行性约束
- 或者仅仅是 `no_local_obstacles_near_free_shape`

`main.py` 会把 `backend_reason` 一并打印出来，便于快速排查优化后端为什么没有生效。

在 `model_based_cem` 控制器中，规划器内部会关闭逐条候选 warning，并把一次规划或一个回合中的优化失败次数、穿透次数和最坏 clearance 汇总到 episode 输出里，避免刷屏。

## 手动控制

```bash
python examples/manual_control_2d.py
```

这个脚本会打开环境窗口和滑块面板，直接调基座推进量和每段左右腱长度增量。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖了这些关键点：

- 2D 正向运动学基础正确
- 腱长差分到曲率的映射正确
- 碰撞响应不会产生 `NaN`
- 目标位姿成功判据正确
- 优化后端能保持分段腱长约束
- 预设接触场景下优化后端不会出现明显穿透
- `reset(seed=...)` 可复现
- `main.py` / `train.py` CLI smoke test 可运行
