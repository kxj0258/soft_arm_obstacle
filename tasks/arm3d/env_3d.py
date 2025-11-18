# tasks/arm3d/env_3d.py
import numpy as np
from gymnasium import spaces
from tasks.base_env import BaseSoftArmEnv
from .sim_3d import SoftArmSim3D   # 上一版 sim_3d.py


class SoftArmObstacle3DEnv(BaseSoftArmEnv):
    """3D 四绳连续体障碍环境（MuJoCo-flex）"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 125}

    def __init__(self, render_mode="human"):
        super().__init__(xml_file="tasks/arm3d/assets/arm_3d_template.xml",
                         render_mode=render_mode)
        self.sim = SoftArmSim3D(self._mujoco_model, self._mujoco_data)

        # 观测 = [x,y,z, qx,qy,qz,qw]  动作 = [dkappa1, dkappa2]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._x_range = (0.4, 1.3)
        self._y_range = (-0.4, 0.4)
        self._z_range = (-0.3, 0.3)

    def _get_obs(self):
        return self.sim.fk().astype(np.float32)

    def _apply_action(self, action: np.ndarray):
        self.sim.step(action)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.sim.arm = self.sim.arm.__class__.default()
        self.sim._update_constraints()
        return self._get_obs(), info

    def _compute_reward(self, obs):
        return -np.linalg.norm(obs[:3] - self.target_pos[:3])