from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from tasks.base_env import BaseSoftArmEnv

from .config import SoftArmObstacle2DEnvConfig
from .model_2d import ArmModel2DConfig
from .renderer_2d import Renderer2D
from .sim_2d import ControlStepTrace2D, Scenario2D, SoftArmSim2D, SoftArmSim2DConfig


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
        self._render_scene_transform: Callable[[dict], dict] | None = None
        self._previous_distance = 0.0
        self._previous_angle_error = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._episode_path_length = 0.0
        self._episode_clearance_sum = 0.0
        self._episode_metric_steps = 0
        self._episode_contact_steps = 0
        self._episode_max_penetration = 0.0
        self._episode_min_clearance = float("inf")
        self._episode_start_distance = 0.0
        self._episode_start_angle_error = 0.0
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
            target_angle=float(self.sim.target_angle),
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
        self._episode_start_angle_error = self._previous_angle_error
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

    def _build_evaluation_metrics(self, *, distance: float, angle_error: float, success: bool) -> dict:
        steps = max(self._episode_metric_steps, 1)
        mean_clearance = self._episode_clearance_sum / steps if self._episode_metric_steps else self.sim.max_reach
        distance_improvement = self._episode_start_distance - float(distance)
        angle_improvement = abs(float(self._episode_start_angle_error)) - abs(float(angle_error))
        path_efficiency = distance_improvement / max(self._episode_path_length, 1e-8)
        return {
            "steps": int(self._episode_metric_steps),
            "distance_improvement": float(distance_improvement),
            "angle_improvement": float(angle_improvement),
            "final_distance": float(distance),
            "final_angle_error": float(angle_error),
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
        self._previous_angle_error = self.sim.angle_error_to_target()
        self._last_reward_terms = {}
        self._reset_episode_metrics()
        observation = self._build_observation()
        info = self._build_info(success=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _build_info(self, *, success: bool) -> dict:
        distance = self.sim.distance_to_target()
        angle_error = self.sim.angle_error_to_target()
        return {
            "distance_to_target": distance,
            "angle_error_to_target": angle_error,
            "collision_count": len(self.sim.model.collision_events),
            "action_blocked": bool(self.sim.model.action_blocked),
            "contact_backend": self.sim.model.contact_backend_used,
            "contact_backend_details": dict(self.sim.model.contact_backend_details),
            "target_position": self.sim.target.copy(),
            "target_angle": float(self.sim.target_angle),
            "target_angle_tolerance": float(self.sim.target_angle_tolerance),
            "tip_position": self.sim.model.tip_position.copy(),
            "tip_angle": float(self.sim.model.tip_angle),
            "base_x": self.sim.model.base_x,
            "kappas": self.sim.model.free_kappas.copy(),
            "tendon_lengths": self.sim.model.free_tendon_lengths.copy(),
            "scenario_name": self.sim.current_scenario.name,
            "contact_summary": self.sim.contact_summary.as_dict(),
            "reward_terms": dict(self._last_reward_terms),
            "evaluation_metrics": self._build_evaluation_metrics(distance=distance, angle_error=angle_error, success=success),
            "action_labels": list(self.action_labels),
            "success": success,
        }

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.control_config.decode_action(action, self.num_kappas)

    def _accumulate_process_reward(self, control_trace: ControlStepTrace2D) -> tuple[float, dict[str, float]]:
        total = 0.0
        accumulated = {
            "time_penalty": 0.0,
            "overlap_penalty": 0.0,
            "process_total": 0.0,
        }
        dt_scale = float(self.sim.sim_config.sim_dt) / max(float(self.sim.sim_config.control_dt), 1e-12)
        for summary in control_trace.micro_contact_summaries:
            process_terms = self.reward_config.compute_process_terms(
                contact_summary=summary,
                max_reach=self.sim.max_reach,
                dt_scale=dt_scale,
            )
            total += float(process_terms["process_total"])
            accumulated["time_penalty"] += float(process_terms["time_penalty"])
            accumulated["overlap_penalty"] += float(process_terms["overlap_penalty"])
            accumulated["process_total"] += float(process_terms["process_total"])
        return float(total), accumulated

    def step(self, action: np.ndarray):
        clipped_action, base_delta, tendon_length_deltas = self.decode_action(action)
        # One control decision is held for `decimation` micro-steps. This
        # decouples low-rate policy updates from high-rate geometric/contact
        # solves while preserving the legacy code path when decimation == 1.
        control_trace = self.sim.step(base_delta=base_delta, tendon_length_deltas=tendon_length_deltas)
        self._elapsed_steps += 1

        distance = self.sim.distance_to_target()
        angle_error = self.sim.angle_error_to_target()
        progress = self._previous_distance - distance
        angle_progress = abs(self._previous_angle_error) - abs(angle_error)
        self._previous_distance = distance
        self._previous_angle_error = angle_error
        success = distance <= self.sim.target_radius and abs(angle_error) <= self.sim.target_angle_tolerance
        self._update_episode_metrics(success=success)

        process_reward, process_terms = self._accumulate_process_reward(control_trace)
        terminal_terms = self.reward_config.compute_terminal_terms(
            action=clipped_action,
            distance=distance,
            progress=progress,
            angle_error=angle_error,
            angle_progress=angle_progress,
            success=success,
            contact_summary=self.sim.contact_summary,
            max_reach=self.sim.max_reach,
        )
        reward = float(process_reward + terminal_terms["terminal_total"])
        self._last_reward_terms = {
            **terminal_terms,
            **process_terms,
            "total": reward,
        }

        terminated = success
        truncated = self._elapsed_steps >= self.max_steps and not terminated
        observation = self._build_observation()
        info = self._build_info(success=success)

        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def set_render_scene_transform(self, transform: Callable[[dict], dict] | None) -> None:
        self._render_scene_transform = transform

    def render(self, scene: dict | None = None):
        if self.render_mode is None:
            raise ValueError("render() requires render_mode='human' or render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = Renderer2D(config=self.render_config)
        scene_data = self.sim.scene_state() if scene is None else dict(scene)
        if self._render_scene_transform is not None:
            scene_data = self._render_scene_transform(scene_data)
        return self._renderer.render(scene_data, mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
