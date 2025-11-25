from typing import List, Tuple
import gymnasium as gym
import numpy as np
from .model_2d import ArmModel2D
from .physics_2d import resolve_impulse
from .renderer_2d import Renderer2D


class SoftArmObstacle2DPyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.obstacles = [(25.0, -5.0, 2.0), (30.0, 5.0, 2.0), (36.0, 2.0, 2.0), (35.0, -5.0, 2.0), (40.0, 10.0, 2.0)]
        self.arm = ArmModel2D(obstacles = self.obstacles)
        self.render_mode = render_mode
        self.renderer = Renderer2D(fps=self.metadata["render_fps"])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        # 动作空间：27 种离散组合
        self.action_space = gym.spaces.Discrete(27)
        self.dt = 1/30
        self.target = (44.0, 11.5)  # 目标位置

    # ---------------- 新增：离散解码 ----------------
    def _decode_action(self, act: int) -> Tuple[float, List[Tuple[float, float]]]:
        """把 0~26 映射成 (dx, dl_vec)"""
        assert 0 <= act < 27, "非法离散动作"
        a0, a1, a2 = act // 9 - 1, (act % 9) // 3 - 1, act % 3 - 1   # 三档各 -1/0/1
        # print("Decoded discrete action:", a0, a1, a2)
        dx = a0 * 0.1                               # 1 mm
        dl_map = {-1: (-0.002,  0.002),   # -1：左收 0.02 mm，右放 0.02 mm
                   0: ( 0.0,     0.0),      # 0：不变
                   1: ( 0.002, -0.002)}   # 1：左放 0.02 mm，右收 0.02 mm
        dl_vec = [dl_map[a1], dl_map[a2]]    # 两段肌腱驱动
        return dx, dl_vec

    def _get_obs(self):
        T = self.arm.T_obs
        return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arm = ArmModel2D(obstacles = self.obstacles)
        # 3. 渲染（可选，只在 human 模式）
        if self.render_mode == "human":
            self.render()       
        return self._get_obs(), {}

    def step(self, action: int):
        dx, dl_vec = self._decode_action(action)
        # print("Decoded action:", action, "-> dx=", dx, " dl_vec=", dl_vec)
        self.arm.apply_action(dx=dx, dl_vec=dl_vec)
        # print("Action applied:", action)

        # 3. 渲染（可选，只在 human 模式）
        if self.render_mode == "human":
            self.render()            
        obs = self._get_obs()
        reward = -np.linalg.norm(obs[:2] - self.target)
        return obs, reward, False, False, {}

    def render(self):
        if self.render_mode == "human":
            self.renderer.update(self.arm, self.obstacles, self.target)
                