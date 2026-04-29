# Soft Arm Obstacle

一个聚焦于**纯 Python 2D 原型**的连续体软臂障碍环境。当前版本不再把 MuJoCo/3D 作为主流程，而是先把 2D PCC 建模、接触求解、Gym 环境、训练脚本、渲染和测试打磨成一条稳定可运行的路线。

## 当前能力

- 2D 分段常曲率软臂建模
- 基座平移 + 多段曲率控制
- 圆障碍采样与目标采样
- 基于碰撞检测的几何接触重构响应
- 参考 OpenTDCRContactModel 的优化式接触后端
- `gymnasium.Env` 接口
- tendon-driven 动作接口
- 可配置的控制/奖励/观测/渲染参数
- 可解释的接触摘要与最近障碍度量
- 参数化的接触响应与轻量接触滞回
- 向量化采样与 broad-phase 加速的碰撞检测
- 绝对 / 相对 / 混合 三种观测模式
- 固定长度的软臂骨架 keypoint 观测
- 分段 clearance / contact mask 观测
- 更适合训练的结构化奖励项
- 与训练奖励分离的评估指标
- 可复现的固定预设场景
- `human` / `rgb_array` 渲染
- PPO 训练与回放脚本
- 自动化测试和 CLI 演示入口

## 快速开始

```bash
conda env create -f environment.yml
conda activate soft_arm

python main.py --render-mode human
```

无界面 smoke test:

```bash
python main.py --render-mode none --episodes 1 --steps 20 --seed 0
```

运行固定预设场景：

```bash
python main.py --render-mode human --scenario-name contact_left
```

切换到优化式接触后端：

```bash
python main.py --render-mode human --scenario-name contact_left --contact-backend optimization
```

训练一个 PPO baseline:

```bash
python train.py --timesteps 20000 --n-envs 4
```

训练时切换观测模式并固定场景：

```bash
python train.py --timesteps 20000 --n-envs 4 --observation-mode hybrid --scenario-name contact_left
```

训练时启用优化式接触后端：

```bash
python train.py --timesteps 20000 --n-envs 1 --scenario-name contact_left --contact-backend optimization
```

回放已训练模型：

```bash
python train.py --play --model-path models/best_model --play-render-mode human
```

快速验证训练链路：

```bash
python train.py --timesteps 16 --eval-freq 0 --n-envs 1 --rollout-steps 16 --max-episode-steps 64 --observation-mode hybrid
```

## 目录结构

```text
soft_arm_obstacle/
├── common/
│   ├── geometry.py        # 2D 向量与姿态工具
│   ├── pcc.py             # 2D PCC 段与连续体链
│   └── utils.py           # 观测拼接等小工具
├── examples/
│   └── manual_control_2d.py
├── tasks/
│   ├── base_env.py
│   ├── arm2d/
│   │   ├── env_2d.py      # Gym 环境
│   │   ├── config.py      # 控制/奖励/观测/渲染配置
│   │   ├── contact_solver_2d.py # OpenTDCR 风格的优化式接触求解
│   │   ├── model_2d.py    # 软臂几何模型
│   │   ├── physics_2d.py  # 碰撞检测与接触响应
│   │   ├── registration.py# Gym 注册辅助
│   │   ├── renderer_2d.py # Matplotlib 渲染
│   │   └── sim_2d.py      # 场景采样与仿真封装
│   └── arm3d/             # 暂缓实现的占位模块
├── tests/
│   ├── test_arm2d_env.py
│   └── test_arm2d_model.py
├── environment.yml
├── main.py
├── train.py
└── seg_circle_tests.py
```

## 环境设计

- 动作：
  `base_motor + per-segment(left_tendon_delta, right_tendon_delta)`
  当前默认是 5 维：
  `[base_motor, seg0_left_delta, seg0_right_delta, seg1_left_delta, seg1_right_delta]`
- 控制映射：
  `delta_base = base_scale * base_motor`
  `delta_l_left/right = tendon_delta_scale * action`
  `kappa_i = (l_right_i - l_left_i) / (2 * tendon_routing_radius * segment_length_i)`
  默认左右腱初始长度为 `10.0`
- 接触后端：
  `heuristic` 保留当前快速几何接触重构；
  `optimization` 使用受 OpenTDCRContactModel 启发的离散曲率约束优化后端；
  若优化解不可行或仍存在穿透，可自动回退到 `heuristic`
- 观测：
  第二阶段支持 `absolute`、`relative`、`hybrid` 三种模式，默认使用 `hybrid`；
  在此基础上还会附加骨架 keypoints、接触摘要和定长分段接触特征
- 奖励：
  由距离改善、归一化距离、动作惩罚、时间惩罚、接触推进奖励、空转接触惩罚、重叠惩罚和成功奖励组成
- 评估指标：
  `path_length`、`path_efficiency`、`mean_clearance`、`contact_ratio`、`max_penetration`、`time_to_goal`
- 终止：
  末端进入目标圆
- 截断：
  达到 `max_steps`

## 第二阶段增强

- `ContactSummary`
  当前步是否处于接触态、接触状态是否为 `collision/latched/free`、最近障碍是谁、最小 clearance 是多少、主要接触 penetration 多大
- `ContactResponse2DConfig`
  接触段长度、采样密度、broad-phase margin、释放 clearance，以及优化式接触后端参数都已参数化
- `Observation2DConfig(mode=...)`
  支持训练更友好的相对特征和混合特征，并可附加骨架 keypoints 与分段接触特征
- `Reward2DConfig`
  奖励项按训练目的拆成多项，并在 `info["reward_terms"]` 中返回
- `info["evaluation_metrics"]`
  训练奖励之外，再单独返回评估指标，便于做 benchmark 和消融
- 预设场景
  可通过 `reset(options={"scenario_name": ...})` 或 CLI 参数复现实验场景
- 训练脚本增强
  `train.py` 支持切换观测模式、骨架点数量、分段接触特征开关、固定预设场景、控制评估频率和评估回合数

## 配置入口

项目第一阶段已经把主要超参数收口到配置对象中：

- `ArmModel2DConfig`
  控制段长、曲率上限、臂半径、最小接触段长度等
- `ContactResponse2DConfig`
  控制 broad-phase 采样、接触长度比例、penetration 增益和接触滞回释放阈值
- `SoftArmSim2DConfig`
  控制障碍和目标的采样范围、数量和间隔
- `SoftArmObstacle2DEnvConfig`
  聚合控制、奖励、观测和渲染配置
- `TendonControl2DConfig`
  控制动作维度、动作范围和单步腱长增量尺度
- `Reward2DConfig`
  控制奖励项权重
- `Observation2DConfig`
  控制观测模式、骨架点数量、是否包含接触特征、分段接触特征、障碍特征、碰撞计数等
- `Render2DConfig`
  控制渲染范围和画布大小

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
        control=TendonControl2DConfig(base_delta_scale=0.8),
        observation=Observation2DConfig(mode="hybrid", backbone_points=5),
        render=Render2DConfig(xlim=(-5.0, 30.0), ylim=(-12.0, 12.0)),
    ),
)
```

`reset(seed=..., options=...)` 还支持传入固定场景：

```python
obs, info = env.reset(
    seed=0,
    options={
        "scenario_name": "contact_left"
    },
)
```

## 运行手动控制示例

```bash
python examples/manual_control_2d.py
```

这个脚本会打开环境窗口和一个滑块面板，方便手动调 `base_x` 和各段曲率动作。

## 训练与评估输出

- `info["reward_terms"]`
  返回训练奖励拆分项
- `info["evaluation_metrics"]`
  返回与训练奖励解耦的评估统计，例如路径长度、平均 clearance、接触占比和最大 penetration
- `info["contact_summary"]`
  返回当前接触状态、最近障碍和分段接触 mask
- `info["contact_backend"]`
  返回当前实际使用的接触后端，便于确认是否走了优化式求解或 heuristic 回退
- `info["tendon_lengths"]`
  返回每段左右腱当前长度，便于检查动作累计效果与曲率来源

## 运行测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖了这些关键点：

- 2D 正向运动学基础正确
- 碰撞响应不会产生 `NaN`
- 接触摘要能报告最近障碍与 clearance
- 接触摘要会返回定长可用的分段接触特征
- 骨架 keypoint 观测可正常生成
- `reset(seed=...)` 可复现
- `reset(options=...)` 支持固定场景
- 预设场景可复现
- 相对观测模式与阶段二奖励项可正常工作
- `evaluation_metrics` 与训练奖励接口可同时工作
- `step()` 接口符合 Gym 约定
- `rgb_array` 渲染返回图像
- `main.py` CLI 可以无界面正常运行
- `train.py --help`、快速训练 smoke test、性能 smoke test 和 Gym 注册辅助可正常工作

## 说明

- `tasks/arm3d/` 目前是明确的暂缓占位实现，避免继续暴露半完成的坏路径。
- 当前同时提供两条接触路径：快速的几何启发式后端，以及受 OpenTDCRContactModel 启发的优化式后端。
- 优化式后端依赖 `scipy`，并且明显慢于 heuristic，更适合做高可信 baseline、调试和小规模训练。
- 如果后续继续扩展，我们建议下一阶段再把 3D 或 MuJoCo 路线单独拉出做第二条主线，而不是再次混入当前 2D 原型。
