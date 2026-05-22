from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .config import Reward3DConfig
from .sim_3d import SoftArmSim3D
from tdcr.actuation import TensionControlConfig


@dataclass(frozen=True)
class ModelBasedCEM3DConfig:
    planning_horizon: int = 5
    num_candidates: int = 32
    num_elites: int = 6
    num_iterations: int = 2
    gamma: float = 0.98
    init_std: float = 0.6
    min_std: float = 0.05
    momentum: float = 0.2
    terminal_distance_weight: float = 0.35
    terminal_orientation_weight: float = 0.25


class ModelBasedCEM3DController:
    def __init__(
        self,
        *,
        sim_template: SoftArmSim3D,
        control_config: TensionControlConfig | None = None,
        reward_config: Reward3DConfig | None = None,
        config: ModelBasedCEM3DConfig | None = None,
        seed: int = 0,
    ):
        self.sim_template = sim_template
        self.control_config = control_config or TensionControlConfig()
        self.reward_config = reward_config or Reward3DConfig()
        self.config = config or ModelBasedCEM3DConfig()
        self.rng = np.random.default_rng(int(seed))
        self._planning_sim: SoftArmSim3D | None = None
        self._warm_start_mean: np.ndarray | None = None
        self._warm_start_std: np.ndarray | None = None
        self.last_plan_info: dict[str, float] = {}

    @property
    def action_dim(self) -> int:
        params = self.sim_template.model_config.robot_params
        return self.control_config.action_dim(params.num_segments, params.tendons_per_segment)

    def reset(self) -> None:
        self._warm_start_mean = None
        self._warm_start_std = None
        self.last_plan_info = {}

    def episode_optimization_event_summary(self) -> dict[str, float]:
        return {
            "event_count": 0.0,
            "fallback_count": 0.0,
            "solver_failure_count": 0.0,
            "penetrating_shape_count": 0.0,
            "worst_min_clearance": float("nan"),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "controller_type": "model_based_cem_3d",
            "config": asdict(self.config),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        sim_template: SoftArmSim3D,
        control_config: TensionControlConfig | None = None,
        reward_config: Reward3DConfig | None = None,
        seed: int = 0,
    ) -> "ModelBasedCEM3DController":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("controller_type") != "model_based_cem_3d":
            raise ValueError(f"Unsupported controller type in {path!s}: {payload.get('controller_type')!r}")
        return cls(
            sim_template=sim_template,
            control_config=control_config,
            reward_config=reward_config,
            config=ModelBasedCEM3DConfig(**payload["config"]),
            seed=seed,
        )

    def act(self, sim: SoftArmSim3D) -> np.ndarray:
        return self.plan_action(sim).astype(np.float32, copy=False)

    def plan_action(self, sim: SoftArmSim3D) -> np.ndarray:
        config = self.config
        horizon = max(int(config.planning_horizon), 1)
        candidates = max(int(config.num_candidates), 1)
        elites = min(max(int(config.num_elites), 1), candidates)
        iterations = max(int(config.num_iterations), 1)
        action_low = float(self.control_config.action_low)
        action_high = float(self.control_config.action_high)

        if self._warm_start_mean is not None and self._warm_start_mean.shape == (horizon, self.action_dim):
            mean = self._warm_start_mean.copy()
            std = self._warm_start_std.copy() if self._warm_start_std is not None else np.full_like(mean, config.init_std)
        else:
            mean = np.zeros((horizon, self.action_dim), dtype=float)
            std = np.full((horizon, self.action_dim), float(config.init_std), dtype=float)

        snapshot = sim.snapshot()
        returns = np.full(candidates, -np.inf, dtype=float)
        elite_returns = np.full(elites, -np.inf, dtype=float)
        for _ in range(iterations):
            samples = self.rng.normal(
                loc=mean[None, :, :],
                scale=np.maximum(std[None, :, :], float(config.min_std)),
                size=(candidates, horizon, self.action_dim),
            )
            samples = np.clip(samples, action_low, action_high)
            for index in range(candidates):
                returns[index] = self.evaluate_action_sequence(snapshot, samples[index])
            elite_indices = np.argsort(returns)[-elites:]
            elite_sequences = samples[elite_indices]
            elite_returns = returns[elite_indices]
            elite_mean = np.mean(elite_sequences, axis=0)
            elite_std = np.std(elite_sequences, axis=0)
            mean = float(config.momentum) * mean + (1.0 - float(config.momentum)) * elite_mean
            std = np.maximum(elite_std, float(config.min_std))

        planned_action = np.clip(mean[0], action_low, action_high).astype(np.float32, copy=False)
        self._warm_start_mean = np.vstack([mean[1:], np.zeros((1, self.action_dim), dtype=float)])
        self._warm_start_std = np.vstack([std[1:], np.full((1, self.action_dim), float(config.init_std), dtype=float)])
        self.last_plan_info = {
            "best_return": float(np.max(elite_returns)),
            "mean_return": float(np.mean(returns)),
            "elite_mean_return": float(np.mean(elite_returns)),
            "planning_horizon": float(horizon),
            "num_candidates": float(candidates),
            "num_iterations": float(iterations),
        }
        return planned_action

    def evaluate_action_sequence(self, snapshot: dict[str, object], action_sequence: np.ndarray) -> float:
        if self._planning_sim is None:
            self._planning_sim = self.sim_template.clone()
        planning_sim = self._planning_sim
        planning_sim.restore(snapshot)

        params = planning_sim.model_config.robot_params
        total_return = 0.0
        discount = 1.0
        previous_distance = planning_sim.distance_to_target()
        previous_orientation_error = planning_sim.orientation_error_to_target()
        for action in np.asarray(action_sequence, dtype=float):
            clipped_action, tau = self.control_config.decode_action(
                action,
                num_segments=params.num_segments,
                tendons_per_segment=params.tendons_per_segment,
                previous_tau=planning_sim.model.tau,
            )
            control_trace = planning_sim.step(tau=tau)
            dt_scale = float(planning_sim.sim_config.sim_dt) / max(float(planning_sim.sim_config.control_dt), 1e-12)
            for summary in control_trace.micro_contact_summaries:
                process_terms = self.reward_config.compute_process_terms(
                    contact_summary=summary,
                    max_reach=planning_sim.max_reach,
                    dt_scale=dt_scale,
                )
                total_return += discount * float(process_terms["process_total"])
            distance = planning_sim.distance_to_target()
            orientation_error = planning_sim.orientation_error_to_target()
            progress = previous_distance - distance
            orientation_progress = abs(previous_orientation_error) - abs(orientation_error)
            success = (
                distance <= planning_sim.target_radius
                and abs(orientation_error) <= planning_sim.target_orientation_tolerance
            )
            reward_terms = self.reward_config.compute_terminal_terms(
                action=clipped_action,
                tau=tau,
                tau_max=self.control_config.tau_max,
                distance=distance,
                progress=progress,
                orientation_error=orientation_error,
                orientation_progress=orientation_progress,
                success=success,
                contact_summary=planning_sim.contact_summary,
                max_reach=planning_sim.max_reach,
            )
            total_return += discount * float(reward_terms["terminal_total"])
            previous_distance = distance
            previous_orientation_error = orientation_error
            if success:
                return float(total_return)
            discount *= float(self.config.gamma)

        reach = max(float(planning_sim.max_reach), 1e-8)
        total_return += discount * (
            -float(self.config.terminal_distance_weight) * float(previous_distance) / reach
            -float(self.config.terminal_orientation_weight) * abs(float(previous_orientation_error)) / np.pi
        )
        return float(total_return)
