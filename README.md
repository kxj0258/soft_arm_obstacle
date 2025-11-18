# 🌈 Soft-Arm-Obstacle  
**连续体软体机械臂在复杂障碍环境中「主动利用碰撞」到达目标位姿的 MuJoCo 仿真框架**  
支持 **2D/3D** 一键切换，碰撞即插入新段，开箱即用。

---

## 📦 功能一览
| 模块 | 描述 |
|------|------|
| ✅ **分段常曲率（PCC）** | 2D 单曲率 / 3D 双曲率（俯仰+偏航） |
| ✅ **绳驱建模** | 2D 左右绳 ∆L→κ；3D 四绳（上下左右）∆L→κ₁,κ₂ |
| ✅ **碰撞-插入** | 与球障碍接触 → 在接触点自动插入新节点（MuJoCo flex） |
| ✅ **标准 gym.Env** | 直接 `sb3.PPO` 训练 |
| ✅ **随机障碍&目标** | 每次 reset 自动生成 |
| ✅ **2D/3D 一键切** | `python main.py --task 2d/3d` |

---

## 🚀 30 秒快速启动

```bash
# 1. 克隆
git clone https://github.com/your_name/soft_arm_obstacle.git
cd soft_arm_obstacle

# 2. 创建 conda 环境（Python 3.10）
conda env create -f environment.yml
conda activate soft_arm

# 3. 运行 2D 演示
python main.py --task 2d
# 或 3D
python main.py --task 3d
```

---

## 🧱 手动安装（无 environment.yml）

```bash
# 创建环境
conda create -n soft_arm python=3.10 -y
conda activate soft_arm

# MuJoCo 官方库 + 渲染依赖
conda install -c conda-forge mujoco>=3.1.0
pip install scipy gymnasium numpy

# （可选）RL 训练
pip install stable-baselines3[extra] tensorboard
```

> 若需 **GPU 渲染**（headless 服务器）再装：
> ```bash
> conda install -c conda-forge mesa-libgl-devel egl
> export MUJOCO_GL=egl
> ```

---

## 📁 工程结构

```
soft_arm_obstacle/
├── common/
│   ├── pcc.py            # PCC 正逆运动学（2D/3D）
│   └── utils.py
├── tasks/
│   ├── base_env.py       # 公共 gym.Env 基类
│   ├── arm2d/
│   │   ├── assets/arm_2d_template.xml
│   │   ├── model_2d.py
│   │   ├── sim_2d.py
│   │   └── env_2d.py
│   └── arm3d/
│       ├── assets/arm_3d_template.xml
│       ├── model_3d.py
│       ├── sim_3d.py
│       └── env_3d.py
├── main.py               # 统一入口
├── environment.yml       # conda 一键环境
└── README.md             # 本文件
```

---

## 🔧 核心 API

| 用法 | 代码 |
|------|------|
| 创建环境 | `env = gym.make("module:SoftArmObstacle2DEnv")` |
| 随机障碍 | 继承 `BaseSoftArmEnv` 自动完成 |
| 自定义目标 | 重写 `_regenerate_obstacles_target()` |
| 插入节点 | 环境已内置，无需手动调用 |

---

## 🎮 运行示例

| 场景 | 命令 |
|------|------|
| 2D 随机测试 | `python main.py --task 2d` |
| 3D 随机测试 | `python main.py --task 3d` |
| RL 训练 2D | `python train.py --task 2d --algo PPO --timesteps 2e6` |
| Headless 渲染 | `export MUJOCO_GL=egl && python main.py --task 3d` |

---

## 📊 训练曲线（示例）

```bash
tensorboard --logdir runs/
# 打开 http://localhost:6006
```

---

## 🛠️ 常见 issue

1. **ImportError: mujoco**  
   → 确保 `conda install -c conda-forge mujoco` 且版本 ≥3.1

2. **GUI 黑屏**  
   → 服务器请 `export MUJOCO_GL=egl` 或 `osmesa`

3. **flex 插入失败**  
   → MuJoCo ≥3.1 才支持 `mj_flexInsert`，升级即可

---

## 🤝 贡献指南

欢迎提 PR / Issue：  
- 添加圆柱、盒子障碍  
- 真实肌腱力-曲率非线性标定  
- 用 Brax/MJX 做可微仿真  

---

## 📜 License

MIT © 2025 Your Name

---

## 📌 Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{soft_arm_obstacle,
  title={Soft-Arm-Obstacle: A MuJoCo-based Framework for Obstacle-aided Continuum Manipulation},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/your_name/soft_arm_obstacle}}
}
```