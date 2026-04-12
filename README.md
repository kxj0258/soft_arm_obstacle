# Soft Arm Obstacle

一个聚焦于**纯 Python 2D 原型**的连续体软臂障碍环境。当前版本不再把 MuJoCo/3D 作为主流程，而是先把 2D PCC 建模、碰撞重构、Gym 环境、渲染和测试打磨成一条稳定可运行的路线。

## 当前能力

- 2D 分段常曲率软臂建模
- 基座平移 + 多段曲率控制
- 圆障碍采样与目标采样
- 基于碰撞检测的“接触重构”响应
- `gymnasium.Env` 接口
- `human` / `rgb_array` 渲染
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
│   │   ├── model_2d.py    # 软臂几何模型
│   │   ├── physics_2d.py  # 碰撞检测与接触响应
│   │   ├── renderer_2d.py # Matplotlib 渲染
│   │   └── sim_2d.py      # 场景采样与仿真封装
│   └── arm3d/             # 暂缓实现的占位模块
├── tests/
│   ├── test_arm2d_env.py
│   └── test_arm2d_model.py
├── environment.yml
└── main.py
```

## 环境设计

- 动作：`[delta_base_x, delta_kappa_0, delta_kappa_1, ...]`
- 观测：
  `tip(x, y, theta) + target(x, y) + base_x + controllable_kappas + flattened_obstacles + collision_count`
- 奖励：
  由目标距离、距离改善量、动作惩罚和轻微接触奖励组成
- 终止：
  末端进入目标圆
- 截断：
  达到 `max_steps`

## 运行手动控制示例

```bash
python examples/manual_control_2d.py
```

这个脚本会打开环境窗口和一个滑块面板，方便手动调 `base_x` 和各段曲率动作。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖了这些关键点：

- 2D 正向运动学基础正确
- 碰撞响应不会产生 `NaN`
- `reset(seed=...)` 可复现
- `step()` 接口符合 Gym 约定
- `rgb_array` 渲染返回图像
- `main.py` CLI 可以无界面正常运行

## 说明

- `tasks/arm3d/` 目前是明确的暂缓占位实现，避免继续暴露半完成的坏路径。
- 当前碰撞响应是面向原型验证的几何启发式方法，不是完整接触力学求解器。
- 如果后续继续扩展，我们建议下一阶段再把 3D 或 MuJoCo 路线单独拉出做第二条主线，而不是再次混入当前 2D 原型。
