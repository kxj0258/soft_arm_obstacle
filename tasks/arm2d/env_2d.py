from __future__ import annotations

import gymnasium as gym
import numpy as np

from common.utils import flatten_obstacles
from tasks.base_env import BaseSoftArmEnv

from .model_2d import ArmModel2DConfig
from .renderer_2d import Renderer2D
from .sim_2d import SoftArmSim2D, SoftArmSim2DConfig


class SoftArmObstacle2DEnv(BaseSoftArmEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_steps: int = 200,
        model_config: ArmModel2DConfig | None = None,
        sim_config: SoftArmSim2DConfig | None = None,
    ):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.sim = SoftArmSim2D(model_config=model_config, sim_config=sim_config)
        self._renderer: Renderer2D | None = None
        self.base_delta_scale = 1.2
        self.curvature_delta_scale = 0.025
        self.progress_reward_weight = 1.5
        self.distance_reward_weight = 0.04
        self.action_penalty_weight = 0.01
        self.contact_bonus = 0.15
        self.success_bonus = 10.0
        self._previous_distance = 0.0

        num_kappas = len(self.sim.model_config.segment_lengths)
        num_obstacles = self.sim.sim_config.obstacle_count
        obs_dim = 3 + 2 + 1 + num_kappas + (3 * num_obstacles) + 1

        self.observation_space = gym.spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1 + num_kappas,), dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        tip_state = self.sim.model.tip_state()
        target = self.sim.target.astype(np.float32)
        base_x = np.array([self.sim.model.base_x], dtype=np.float32)
        kappas = self.sim.model.free_kappas.astype(np.float32)
        obstacles = flatten_obstacles(self.sim.obstacles)
        collision_count = np.array([len(self.sim.model.collision_events)], dtype=np.float32)
        return np.concatenate([tip_state, target, base_x, kappas, obstacles, collision_count]).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.sim.reset(self.np_random)
        self._previous_distance = self.sim.distance_to_target()
        observation = self._build_observation()
        info = self._build_info(success=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _build_info(self, *, success: bool) -> dict:
        return {
            "distance_to_target": self.sim.distance_to_target(),
            "collision_count": len(self.sim.model.collision_events),
            "target_position": self.sim.target.copy(),
            "tip_position": self.sim.model.tip_position.copy(),
            "success": success,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        base_delta = float(action[0]) * self.base_delta_scale
        kappa_delta = action[1:] * self.curvature_delta_scale
        self.sim.step(base_delta=base_delta, kappa_delta=kappa_delta)
        self._elapsed_steps += 1

        distance = self.sim.distance_to_target()
        progress = self._previous_distance - distance
        self._previous_distance = distance
        success = distance <= self.sim.target_radius

        reward = (
            self.progress_reward_weight * progress
            - self.distance_reward_weight * distance
            - self.action_penalty_weight * float(np.linalg.norm(action))
            + self.contact_bonus * min(1, len(self.sim.model.collision_events))
        )
        if success:
            reward += self.success_bonus

        terminated = success
        truncated = self._elapsed_steps >= self.max_steps and not terminated
        observation = self._build_observation()
        info = self._build_info(success=success)

        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            raise ValueError("render() requires render_mode='human' or render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = Renderer2D()
        return self._renderer.render(self.sim.scene_state(), mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
                
