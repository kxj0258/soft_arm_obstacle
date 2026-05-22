from __future__ import annotations

import gymnasium as gym
import numpy as np

from tasks.base_env import BaseSoftArmEnv

from .config import SoftArmObstacle3DEnvConfig
from .model_3d import ArmModel3DConfig
from .renderer_3d import Renderer3D
from .sim_3d import ControlStepTrace3D, Scenario3D, SoftArmSim3D, SoftArmSim3DConfig


class SoftArmObstacle3DEnv(BaseSoftArmEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_steps: int = 200,
        env_config: SoftArmObstacle3DEnvConfig | None = None,
        model_config: ArmModel3DConfig | None = None,
        sim_config: SoftArmSim3DConfig | None = None,
    ):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.sim = SoftArmSim3D(model_config=model_config, sim_config=sim_config)
        self.env_config = env_config or SoftArmObstacle3DEnvConfig()
        self.control_config = self.env_config.control
        self.reward_config = self.env_config.reward
        self.observation_config = self.env_config.observation
        self.render_config = self.env_config.render
        self._renderer: Renderer3D | None = None
        self._previous_distance = 0.0
        self._previous_orientation_error = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._episode_path_length = 0.0
        self._episode_clearance_sum = 0.0
        self._episode_metric_steps = 0
        self._episode_contact_steps = 0
        self._episode_max_penetration = 0.0
        self._episode_min_clearance = float("inf")
        self._episode_start_distance = 0.0
        self._episode_start_orientation_error = 0.0
        self._episode_success_step: int | None = None
        self._last_tip_position = np.zeros(3, dtype=float)

        self.num_segments = self.sim.model_config.robot_params.num_segments
        self.tendons_per_segment = self.sim.model_config.robot_params.tendons_per_segment
        self.tau_dim = self.sim.model_config.robot_params.tau_dim
        self.max_obstacles = self.sim.sim_config.obstacle_count
        self.action_labels = self.control_config.action_labels(self.num_segments, self.tendons_per_segment)
        obs_dim = self.observation_config.observation_dim(
            tau_dim=self.tau_dim,
            num_segments=self.num_segments,
            max_obstacles=self.max_obstacles,
        )

        self.observation_space = gym.spaces.Box(
            low=self.observation_config.low,
            high=self.observation_config.high,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        action_dim = self.control_config.action_dim(self.num_segments, self.tendons_per_segment)
        self.action_space = gym.spaces.Box(
            low=self.control_config.action_low,
            high=self.control_config.action_high,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _build_observation(self) -> np.ndarray:
        return self.observation_config.build_observation(
            tip_position=self.sim.model.tip_position,
            tip_rotvec=self.sim.model.tip_rotvec,
            target=self.sim.target.astype(np.float32),
            target_rotvec=self.sim.target_rotvec.astype(np.float32),
            tau=self.sim.model.tau.astype(np.float32),
            tau_max=self.control_config.tau_max,
            strain_u_by_segment=self.sim.model.strain_u_by_segment().astype(np.float32),
            obstacles=self.sim.obstacles,
            contact_summary=self.sim.contact_summary,
            backbone_points=self.sim.model.backbone_points(self.observation_config.backbone_points),
            collision_count=self.sim.contact_summary.collision_count,
            max_obstacles=self.max_obstacles,
            max_reach=self.sim.max_reach,
            curvature_limit=self.sim.model_config.robot_params.curvature_limit,
        )

    def _reset_episode_metrics(self) -> None:
        self._episode_path_length = 0.0
        self._episode_clearance_sum = 0.0
        self._episode_metric_steps = 0
        self._episode_contact_steps = 0
        self._episode_max_penetration = 0.0
        self._episode_min_clearance = float("inf")
        self._episode_start_distance = self._previous_distance
        self._episode_start_orientation_error = self._previous_orientation_error
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

    def _build_evaluation_metrics(self, *, distance: float, orientation_error: float, success: bool) -> dict:
        steps = max(self._episode_metric_steps, 1)
        mean_clearance = self._episode_clearance_sum / steps if self._episode_metric_steps else self.sim.max_reach
        distance_improvement = self._episode_start_distance - float(distance)
        orientation_improvement = abs(float(self._episode_start_orientation_error)) - abs(float(orientation_error))
        path_efficiency = distance_improvement / max(self._episode_path_length, 1e-8)
        return {
            "steps": int(self._episode_metric_steps),
            "distance_improvement": float(distance_improvement),
            "orientation_improvement": float(orientation_improvement),
            "final_distance": float(distance),
            "final_orientation_error": float(orientation_error),
            "path_length": float(self._episode_path_length),
            "path_efficiency": float(path_efficiency),
            "mean_clearance": float(mean_clearance),
            "min_clearance": float(
                self._episode_min_clearance if np.isfinite(self._episode_min_clearance) else self.sim.max_reach
            ),
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
        self._previous_orientation_error = self.sim.orientation_error_to_target()
        self._last_reward_terms = {}
        self._reset_episode_metrics()
        observation = self._build_observation()
        info = self._build_info(success=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _build_info(self, *, success: bool) -> dict:
        distance = self.sim.distance_to_target()
        orientation_error = self.sim.orientation_error_to_target()
        return {
            "distance_to_target": distance,
            "orientation_error_to_target": orientation_error,
            "collision_count": int(self.sim.contact_summary.collision_count),
            "contact_backend": self.sim.model.contact_backend_used,
            "contact_backend_details": dict(self.sim.model.contact_backend_details),
            "target_position": self.sim.target.copy(),
            "target_rotvec": self.sim.target_rotvec.copy(),
            "target_radius": float(self.sim.target_radius),
            "target_orientation_tolerance": float(self.sim.target_orientation_tolerance),
            "tip_position": self.sim.model.tip_position.copy(),
            "tip_rotvec": self.sim.model.tip_rotvec.copy(),
            "tau": self.sim.model.tau.copy(),
            "strain_u_by_segment": self.sim.model.strain_u_by_segment().copy(),
            "scenario_name": self.sim.current_scenario.name,
            "contact_summary": self.sim.contact_summary.as_dict(),
            "reward_terms": dict(self._last_reward_terms),
            "evaluation_metrics": self._build_evaluation_metrics(
                distance=distance,
                orientation_error=orientation_error,
                success=success,
            ),
            "action_labels": list(self.action_labels),
            "success": success,
        }

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.control_config.decode_action(
            action,
            num_segments=self.num_segments,
            tendons_per_segment=self.tendons_per_segment,
            previous_tau=self.sim.model.tau,
        )

    def _accumulate_process_reward(self, control_trace: ControlStepTrace3D) -> tuple[float, dict[str, float]]:
        total = 0.0
        accumulated = {
            "time_penalty": 0.0,
            "penetration_penalty": 0.0,
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
            accumulated["penetration_penalty"] += float(process_terms["penetration_penalty"])
            accumulated["process_total"] += float(process_terms["process_total"])
        return float(total), accumulated

    def step(self, action: np.ndarray):
        clipped_action, tau = self.decode_action(action)
        # The policy acts at `control_dt` while the shape/contact solver can be
        # evaluated at `sim_dt`. A first-order hold interpolates the requested
        # actuator target over the inner simulation loop.
        control_trace = self.sim.step(tau=tau)
        self._elapsed_steps += 1

        distance = self.sim.distance_to_target()
        orientation_error = self.sim.orientation_error_to_target()
        progress = self._previous_distance - distance
        orientation_progress = abs(self._previous_orientation_error) - abs(orientation_error)
        self._previous_distance = distance
        self._previous_orientation_error = orientation_error
        success = (
            distance <= self.sim.target_radius
            and abs(orientation_error) <= self.sim.target_orientation_tolerance
        )
        self._update_episode_metrics(success=success)

        process_reward, process_terms = self._accumulate_process_reward(control_trace)
        terminal_terms = self.reward_config.compute_terminal_terms(
            action=clipped_action,
            tau=tau,
            tau_max=self.control_config.tau_max,
            distance=distance,
            progress=progress,
            orientation_error=orientation_error,
            orientation_progress=orientation_progress,
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

    def render(self):
        if self.render_mode is None:
            raise ValueError("render() requires render_mode='human' or render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = Renderer3D(config=self.render_config)
        return self._renderer.render(self.sim.scene_state(), mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
