from __future__ import annotations

import gymnasium as gym
import numpy as np

from tasks.base_env import BaseSoftArmEnv

from .config import SoftArmObstacle2DEnvConfig
from .model_2d import ArmModel2DConfig
from .renderer_2d import Renderer2D
from .sim_2d import Scenario2D, SoftArmSim2D, SoftArmSim2DConfig


class SoftArmObstacle2DEnv(BaseSoftArmEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_steps: int = 200,
        env_config: SoftArmObstacle2DEnvConfig | None = None,
        model_config: ArmModel2DConfig | None = None,
        sim_config: SoftArmSim2DConfig | None = None,
    ):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.sim = SoftArmSim2D(model_config=model_config, sim_config=sim_config)
        self.env_config = env_config or SoftArmObstacle2DEnvConfig()
        self.control_config = self.env_config.control
        self.reward_config = self.env_config.reward
        self.observation_config = self.env_config.observation
        self.render_config = self.env_config.render
        self._renderer: Renderer2D | None = None
        self._previous_distance = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._episode_path_length = 0.0
        self._episode_clearance_sum = 0.0
        self._episode_metric_steps = 0
        self._episode_contact_steps = 0
        self._episode_max_penetration = 0.0
        self._episode_min_clearance = float("inf")
        self._episode_start_distance = 0.0
        self._episode_success_step: int | None = None
        self._last_tip_position = np.zeros(2, dtype=float)

        self.num_kappas = len(self.sim.model_config.segment_lengths)
        self.max_obstacles = self.sim.sim_config.obstacle_count
        self.action_labels = self.control_config.action_labels(self.num_kappas)
        obs_dim = self.observation_config.observation_dim(
            num_kappas=self.num_kappas,
            max_obstacles=self.max_obstacles,
        )

        self.observation_space = gym.spaces.Box(
            low=self.observation_config.low,
            high=self.observation_config.high,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        action_dim = self.control_config.action_dim(self.num_kappas)
        self.action_space = gym.spaces.Box(
            low=self.control_config.action_low,
            high=self.control_config.action_high,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _build_observation(self) -> np.ndarray:
        return self.observation_config.build_observation(
            tip_state=self.sim.model.tip_state(),
            target=self.sim.target.astype(np.float32),
            base_x=self.sim.model.base_x,
            kappas=self.sim.model.free_kappas.astype(np.float32),
            obstacles=self.sim.obstacles,
            contact_summary=self.sim.contact_summary,
            backbone_points=self.sim.model.backbone_points(self.observation_config.backbone_points),
            collision_count=len(self.sim.model.collision_events),
            max_obstacles=self.max_obstacles,
            max_reach=self.sim.max_reach,
            base_x_limits=self.sim.model_config.base_x_limits,
            curvature_limit=self.sim.model_config.curvature_limit,
        )

    def _reset_episode_metrics(self) -> None:
        self._episode_path_length = 0.0
        self._episode_clearance_sum = 0.0
        self._episode_metric_steps = 0
        self._episode_contact_steps = 0
        self._episode_max_penetration = 0.0
        self._episode_min_clearance = float("inf")
        self._episode_start_distance = self._previous_distance
        self._episode_success_step = None
        self._last_tip_position = self.sim.model.tip_position.copy()

    def _update_episode_metrics(self, *, success: bool) -> None:
        current_tip = self.sim.model.tip_position.copy()
        self._episode_path_length += float(np.linalg.norm(current_tip - self._last_tip_position))
        self._last_tip_position = current_tip

        current_clearance = float(self.sim.contact_summary.min_clearance)
        if not np.isfinite(current_clearance):
            current_clearance = self.sim.max_reach
        self._episode_clearance_sum += current_clearance
        self._episode_metric_steps += 1
        self._episode_contact_steps += int(self.sim.contact_summary.active)
        self._episode_max_penetration = max(
            self._episode_max_penetration,
            float(self.sim.contact_summary.primary_penetration),
        )
        self._episode_min_clearance = min(self._episode_min_clearance, current_clearance)
        if success and self._episode_success_step is None:
            self._episode_success_step = self._elapsed_steps

    def _build_evaluation_metrics(self, *, distance: float, success: bool) -> dict:
        steps = max(self._episode_metric_steps, 1)
        mean_clearance = self._episode_clearance_sum / steps if self._episode_metric_steps else self.sim.max_reach
        distance_improvement = self._episode_start_distance - float(distance)
        path_efficiency = distance_improvement / max(self._episode_path_length, 1e-8)
        return {
            "steps": int(self._episode_metric_steps),
            "distance_improvement": float(distance_improvement),
            "final_distance": float(distance),
            "path_length": float(self._episode_path_length),
            "path_efficiency": float(path_efficiency),
            "mean_clearance": float(mean_clearance),
            "min_clearance": float(self._episode_min_clearance if np.isfinite(self._episode_min_clearance) else self.sim.max_reach),
            "contact_ratio": float(self._episode_contact_steps / steps),
            "max_penetration": float(self._episode_max_penetration),
            "time_to_goal": float(self._episode_success_step if self._episode_success_step is not None else -1.0),
            "success": bool(success),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        if options is None:
            scenario = None
        else:
            scenario = options.get("scenario")
            if scenario is None and "scenario_name" in options:
                scenario = {"scenario_name": options["scenario_name"]}
            if scenario is None:
                scenario = options
        self.sim.reset(self.np_random, scenario=scenario)
        self._previous_distance = self.sim.distance_to_target()
        self._last_reward_terms = {}
        self._reset_episode_metrics()
        observation = self._build_observation()
        info = self._build_info(success=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _build_info(self, *, success: bool) -> dict:
        distance = self.sim.distance_to_target()
        return {
            "distance_to_target": distance,
            "collision_count": len(self.sim.model.collision_events),
            "contact_backend": self.sim.model.contact_backend_used,
            "target_position": self.sim.target.copy(),
            "tip_position": self.sim.model.tip_position.copy(),
            "base_x": self.sim.model.base_x,
            "kappas": self.sim.model.free_kappas.copy(),
            "tendon_lengths": self.sim.model.free_tendon_lengths.copy(),
            "scenario_name": self.sim.current_scenario.name,
            "contact_summary": self.sim.contact_summary.as_dict(),
            "reward_terms": dict(self._last_reward_terms),
            "evaluation_metrics": self._build_evaluation_metrics(distance=distance, success=success),
            "action_labels": list(self.action_labels),
            "success": success,
        }

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.control_config.decode_action(action, self.num_kappas)

    def step(self, action: np.ndarray):
        clipped_action, base_delta, tendon_length_deltas = self.decode_action(action)
        self.sim.step(base_delta=base_delta, tendon_length_deltas=tendon_length_deltas)
        self._elapsed_steps += 1

        distance = self.sim.distance_to_target()
        progress = self._previous_distance - distance
        self._previous_distance = distance
        success = distance <= self.sim.target_radius
        self._update_episode_metrics(success=success)

        reward_terms = self.reward_config.compute_terms(
            action=clipped_action,
            distance=distance,
            progress=progress,
            success=success,
            contact_summary=self.sim.contact_summary,
            max_reach=self.sim.max_reach,
        )
        reward = reward_terms["total"]
        self._last_reward_terms = reward_terms

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
            self._renderer = Renderer2D(config=self.render_config)
        return self._renderer.render(self.sim.scene_state(), mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
