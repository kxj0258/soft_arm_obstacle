import gymnasium as gym
import numpy as np
from .model_2d import ArmModel2D
from .physics_2d import resolve_impulse
from .renderer_2d import Renderer2D


class SoftArmObstacle2DPyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.obstacles = [(25.0, -5.0, 3.0), (40.0, 10.0, 3.0)]
        self.arm = ArmModel2D(obstacles = self.obstacles)
        self.render_mode = render_mode
        self.renderer = Renderer2D(fps=self.metadata["render_fps"])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.dt = 1/30
        self.target = (50.0, 15.0)  # 目标位置

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

    def step(self, action):
        self.arm.apply_action(dx=action[0]*1.0,
                              dkappa=action[1:]*0.01)
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
                