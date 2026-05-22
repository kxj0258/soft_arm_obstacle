from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np

from .config import Reward2DConfig, TendonControl2DConfig
from .sim_2d import SoftArmSim2D

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBasedCEMConfig:
    planning_horizon: int = 6
    num_candidates: int = 48
    num_elites: int = 8
    num_iterations: int = 3
    gamma: float = 0.98
    init_std: float = 0.6
    min_std: float = 0.05
    momentum: float = 0.2
    terminal_distance_weight: float = 0.35
    terminal_angle_weight: float = 0.25


class ModelBasedCEMController:
    def __init__(
        self,
        *,
        sim_template: SoftArmSim2D,
        control_config: TendonControl2DConfig | None = None,
        reward_config: Reward2DConfig | None = None,
        config: ModelBasedCEMConfig | None = None,
        seed: int = 0,
    ):
        self.sim_template = sim_template
        self.control_config = control_config or TendonControl2DConfig()
        self.reward_config = reward_config or Reward2DConfig()
        self.config = config or ModelBasedCEMConfig()
        self.rng = np.random.default_rng(int(seed))
        self._planning_sim: SoftArmSim2D | None = None
        self._warm_start_mean: np.ndarray | None = None
        self._warm_start_std: np.ndarray | None = None
        self.last_plan_info: dict[str, float] = {}
        self._episode_optimization_event_summary = self._make_empty_optimization_event_summary()
        self._last_plan_optimization_event_summary = self._make_empty_optimization_event_summary()

    @property
    def action_dim(self) -> int:
        return self.control_config.action_dim(len(self.sim_template.model_config.segment_lengths))

    def reset(self) -> None:
        self._warm_start_mean = None
        self._warm_start_std = None
        self.last_plan_info = {}
        self._episode_optimization_event_summary = self._make_empty_optimization_event_summary()
        self._last_plan_optimization_event_summary = self._make_empty_optimization_event_summary()

    def _make_empty_optimization_event_summary(self) -> dict[str, float]:
        return {
            "event_count": 0.0,
            "fallback_count": 0.0,
            "solver_failure_count": 0.0,
            "penetrating_shape_count": 0.0,
            "worst_min_clearance": float("inf"),
        }

    def _merge_optimization_event_summary(self, target: dict[str, float], update: dict[str, object]) -> dict[str, float]:
        target["event_count"] += float(update.get("event_count", 0.0))
        target["fallback_count"] += float(update.get("fallback_count", 0.0))
        target["solver_failure_count"] += float(update.get("solver_failure_count", 0.0))
        target["penetrating_shape_count"] += float(update.get("penetrating_shape_count", 0.0))
        update_worst = float(update.get("worst_min_clearance", float("inf")))
        target["worst_min_clearance"] = min(float(target["worst_min_clearance"]), update_worst)
        return target

    def episode_optimization_event_summary(self) -> dict[str, float]:
        summary = dict(self._episode_optimization_event_summary)
        if not np.isfinite(summary["worst_min_clearance"]):
            summary["worst_min_clearance"] = float("nan")
        return summary

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "controller_type": "model_based_cem",
            "config": asdict(self.config),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        sim_template: SoftArmSim2D,
        control_config: TendonControl2DConfig | None = None,
        reward_config: Reward2DConfig | None = None,
        seed: int = 0,
    ) -> "ModelBasedCEMController":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("controller_type") != "model_based_cem":
            raise ValueError(f"Unsupported controller type in {path!s}: {payload.get('controller_type')!r}")
        return cls(
            sim_template=sim_template,
            control_config=control_config,
            reward_config=reward_config,
            config=ModelBasedCEMConfig(**payload["config"]),
            seed=seed,
        )

    def act(self, sim: SoftArmSim2D) -> np.ndarray:
        action = self.plan_action(sim)
        return action.astype(np.float32, copy=False)

    def plan_action(self, sim: SoftArmSim2D) -> np.ndarray:
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
        plan_summaries = [self._make_empty_optimization_event_summary() for _ in range(candidates)]
        planning_cycle_summary = self._make_empty_optimization_event_summary()
        elite_returns = np.full(elites, -np.inf, dtype=float)
        for _ in range(iterations):
            samples = self.rng.normal(
                loc=mean[None, :, :],
                scale=np.maximum(std[None, :, :], float(config.min_std)),
                size=(candidates, horizon, self.action_dim),
            )
            samples = np.clip(samples, action_low, action_high)
            for index in range(candidates):
                returns[index], plan_summaries[index] = self.evaluate_action_sequence(snapshot, samples[index])
                self._merge_optimization_event_summary(planning_cycle_summary, plan_summaries[index])
            elite_indices = np.argsort(returns)[-elites:]
            elite_sequences = samples[elite_indices]
            elite_returns = returns[elite_indices]
            elite_mean = np.mean(elite_sequences, axis=0)
            elite_std = np.std(elite_sequences, axis=0)
            mean = float(config.momentum) * mean + (1.0 - float(config.momentum)) * elite_mean
            std = np.maximum(elite_std, float(config.min_std))

        planned_action = np.clip(mean[0], action_low, action_high).astype(np.float32, copy=False)
        shifted_mean = np.vstack([mean[1:], np.zeros((1, self.action_dim), dtype=float)])
        shifted_std = np.vstack([std[1:], np.full((1, self.action_dim), float(config.init_std), dtype=float)])
        self._warm_start_mean = shifted_mean
        self._warm_start_std = shifted_std
        self._last_plan_optimization_event_summary = dict(planning_cycle_summary)
        self._merge_optimization_event_summary(
            self._episode_optimization_event_summary,
            self._last_plan_optimization_event_summary,
        )
        worst_min_clearance = float(self._last_plan_optimization_event_summary["worst_min_clearance"])
        if not np.isfinite(worst_min_clearance):
            worst_min_clearance = float("nan")
        self.last_plan_info = {
            "best_return": float(np.max(elite_returns)),
            "mean_return": float(np.mean(returns)),
            "elite_mean_return": float(np.mean(elite_returns)),
            "planning_horizon": float(horizon),
            "num_candidates": float(candidates),
            "num_iterations": float(iterations),
            "optimization_event_count": float(self._last_plan_optimization_event_summary["event_count"]),
            "optimization_solver_failure_count": float(self._last_plan_optimization_event_summary["solver_failure_count"]),
            "optimization_penetrating_shape_count": float(self._last_plan_optimization_event_summary["penetrating_shape_count"]),
            "optimization_worst_min_clearance": worst_min_clearance,
        }
        return planned_action

    def evaluate_action_sequence(self, snapshot: dict[str, object], action_sequence: np.ndarray) -> tuple[float, dict[str, float]]:
        if self._planning_sim is None:
            self._planning_sim = self.sim_template.clone()
            self._planning_sim.model.set_optimization_event_logging(False)
        planning_sim = self._planning_sim
        planning_sim.restore(snapshot)
        planning_sim.model.reset_optimization_event_summary()
        sequence_summary = self._make_empty_optimization_event_summary()

        total_return = 0.0
        discount = 1.0
        previous_distance = planning_sim.distance_to_target()
        previous_angle_error = planning_sim.angle_error_to_target()
        for action in np.asarray(action_sequence, dtype=float):
            clipped_action, base_delta, tendon_length_deltas = self.control_config.decode_action(
                action,
                len(planning_sim.model_config.segment_lengths),
            )
            control_trace = planning_sim.step(base_delta=base_delta, tendon_length_deltas=tendon_length_deltas)
            self._merge_optimization_event_summary(
                sequence_summary,
                planning_sim.model.consume_optimization_event_summary(),
            )
            dt_scale = float(planning_sim.sim_config.sim_dt) / max(float(planning_sim.sim_config.control_dt), 1e-12)
            for summary in control_trace.micro_contact_summaries:
                process_terms = self.reward_config.compute_process_terms(
                    contact_summary=summary,
                    max_reach=planning_sim.max_reach,
                    dt_scale=dt_scale,
                )
                total_return += discount * float(process_terms["process_total"])
            distance = planning_sim.distance_to_target()
            angle_error = planning_sim.angle_error_to_target()
            progress = previous_distance - distance
            angle_progress = abs(previous_angle_error) - abs(angle_error)
            success = (
                distance <= planning_sim.target_radius
                and abs(angle_error) <= planning_sim.target_angle_tolerance
            )
            reward_terms = self.reward_config.compute_terminal_terms(
                action=clipped_action,
                distance=distance,
                progress=progress,
                angle_error=angle_error,
                angle_progress=angle_progress,
                success=success,
                contact_summary=planning_sim.contact_summary,
                max_reach=planning_sim.max_reach,
            )
            total_return += discount * float(reward_terms["terminal_total"])
            previous_distance = distance
            previous_angle_error = angle_error
            if success:
                return float(total_return), sequence_summary
            discount *= float(self.config.gamma)

        reach = max(float(planning_sim.max_reach), 1e-8)
        total_return += discount * (
            -float(self.config.terminal_distance_weight) * float(previous_distance) / reach
            -float(self.config.terminal_angle_weight) * abs(float(previous_angle_error)) / np.pi
        )
        return float(total_return), sequence_summary
