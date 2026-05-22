from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.frames import rotvec_from_matrix
from tdcr.shapes import ContactSummary3D

from .model_3d import ArmModel3D, ArmModel3DConfig
from .physics_3d import Obstacle3D, ensure_obstacles


@dataclass(frozen=True)
class SoftArmSim3DConfig:
    obstacle_count: int = 5
    obstacle_radius_range: tuple[float, float] = (0.012, 0.026)
    obstacle_x_range: tuple[float, float] = (-0.075, 0.075)
    obstacle_y_range: tuple[float, float] = (-0.075, 0.075)
    obstacle_z_range: tuple[float, float] = (0.035, 0.215)
    target_x_range: tuple[float, float] = (-0.075, 0.075)
    target_y_range: tuple[float, float] = (-0.075, 0.075)
    target_z_range: tuple[float, float] = (0.15, 0.24)
    target_rotvec_range: tuple[float, float] = (-0.75, 0.75)
    target_radius: float = 0.012
    target_orientation_tolerance: float = 0.45
    min_obstacle_gap: float = 0.008
    min_target_gap: float = 0.012
    base_clearance: float = 0.018
    max_sampling_attempts: int = 512
    sim_dt: float = 0.001
    control_dt: float = 0.02

    def __post_init__(self) -> None:
        sim_dt = float(self.sim_dt)
        control_dt = float(self.control_dt)
        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive.")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive.")
        if control_dt + 1e-12 < sim_dt:
            raise ValueError("control_dt must be greater than or equal to sim_dt.")
        ratio = control_dt / sim_dt
        rounded = int(round(ratio))
        if rounded < 1 or not np.isclose(ratio, float(rounded), atol=1e-9, rtol=1e-6):
            raise ValueError(
                "control_dt / sim_dt must be an integer decimation factor. "
                f"Received sim_dt={sim_dt} and control_dt={control_dt}."
            )
        object.__setattr__(self, "sim_dt", sim_dt)
        object.__setattr__(self, "control_dt", control_dt)

    @property
    def decimation(self) -> int:
        return int(round(float(self.control_dt) / max(float(self.sim_dt), 1e-12)))


@dataclass(frozen=True)
class ControlStepTrace3D:
    sim_dt: float
    control_dt: float
    decimation: int
    micro_contact_summaries: tuple[ContactSummary3D, ...]
    contact_summary: ContactSummary3D


@dataclass
class Scenario3D:
    name: str | None = None
    obstacles: list[Obstacle3D | dict | tuple | list] | None = None
    target: np.ndarray | tuple[float, float, float] | None = None
    target_rotvec: np.ndarray | tuple[float, float, float] | None = None
    tau: np.ndarray | None = None


PRESET_SCENARIOS: dict[str, Scenario3D] = {
    "reach_easy_3d": Scenario3D(
        name="reach_easy_3d",
        obstacles=[],
        target=(0.0, 0.0, 0.24),
        target_rotvec=(0.0, 0.0, 0.0),
        tau=np.zeros((3, 3), dtype=float),
    ),
    "contact_sphere_3d": Scenario3D(
        name="contact_sphere_3d",
        obstacles=[(0.018, 0.0, 0.12, 0.020)],
        target=(-0.015, 0.025, 0.22),
        target_rotvec=(0.0, 0.0, 0.0),
        tau=np.zeros((3, 3), dtype=float),
    ),
    "channel_3d": Scenario3D(
        name="channel_3d",
        obstacles=[
            (0.035, 0.0, 0.08, 0.018),
            (-0.032, 0.020, 0.14, 0.020),
            (0.030, -0.025, 0.19, 0.018),
        ],
        target=(0.0, 0.035, 0.23),
        target_rotvec=(0.15, 0.0, 0.0),
        tau=np.zeros((3, 3), dtype=float),
    ),
}


def clone_scenario(scenario: Scenario3D) -> Scenario3D:
    return Scenario3D(
        name=scenario.name,
        obstacles=None if scenario.obstacles is None else ensure_obstacles(scenario.obstacles),
        target=None if scenario.target is None else np.asarray(scenario.target, dtype=float).copy(),
        target_rotvec=None
        if scenario.target_rotvec is None
        else np.asarray(scenario.target_rotvec, dtype=float).reshape(3).copy(),
        tau=None if scenario.tau is None else np.asarray(scenario.tau, dtype=float).copy(),
    )


def list_preset_scenarios() -> tuple[str, ...]:
    return tuple(PRESET_SCENARIOS.keys())


def get_preset_scenario(name: str) -> Scenario3D:
    if name not in PRESET_SCENARIOS:
        available = ", ".join(list_preset_scenarios())
        raise KeyError(f"Unknown 3D preset scenario {name!r}. Available presets: {available}")
    return clone_scenario(PRESET_SCENARIOS[name])


class SoftArmSim3D:
    def __init__(
        self,
        model_config: ArmModel3DConfig | None = None,
        sim_config: SoftArmSim3DConfig | None = None,
    ):
        self.model_config = model_config or ArmModel3DConfig()
        self.sim_config = sim_config or SoftArmSim3DConfig()
        self.model = ArmModel3D(config=self.model_config)
        self.obstacles: list[Obstacle3D] = []
        self.target = np.zeros(3, dtype=float)
        self.target_rotvec = np.zeros(3, dtype=float)
        self.target_radius = float(self.sim_config.target_radius)
        self.target_orientation_tolerance = float(self.sim_config.target_orientation_tolerance)
        self.current_scenario = Scenario3D()
        self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace: ControlStepTrace3D | None = None

    @property
    def max_reach(self) -> float:
        return float(self.model_config.robot_params.total_length)

    @property
    def decimation(self) -> int:
        return self.sim_config.decimation

    def _workspace_radius(self, z: float) -> float:
        return max(self.max_reach - float(z), 0.0) + 0.08

    def _inside_workspace(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float).reshape(3)
        if point[2] < -1e-9 or point[2] > self.max_reach + 1e-9:
            return False
        radial = float(np.linalg.norm(point[:2]))
        return radial <= self._workspace_radius(point[2])

    def _sample_obstacles(self, rng: np.random.Generator) -> list[Obstacle3D]:
        obstacles: list[Obstacle3D] = []
        for _ in range(self.sim_config.obstacle_count):
            for _ in range(self.sim_config.max_sampling_attempts):
                radius = float(rng.uniform(*self.sim_config.obstacle_radius_range))
                center = np.array(
                    [
                        rng.uniform(*self.sim_config.obstacle_x_range),
                        rng.uniform(*self.sim_config.obstacle_y_range),
                        rng.uniform(*self.sim_config.obstacle_z_range),
                    ],
                    dtype=float,
                )
                if np.linalg.norm(center) < self.sim_config.base_clearance + radius:
                    continue
                if not self._inside_workspace(center):
                    continue
                if any(
                    np.linalg.norm(center - obstacle.center)
                    < (radius + obstacle.radius + self.sim_config.min_obstacle_gap)
                    for obstacle in obstacles
                ):
                    continue
                obstacles.append(Obstacle3D(center=center, radii=radius, kind="sphere"))
                break
            else:
                raise RuntimeError("Failed to sample a collision-free 3D obstacle layout.")
        return obstacles

    def _sample_target_pose(
        self,
        rng: np.random.Generator,
        obstacles: list[Obstacle3D],
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(self.sim_config.max_sampling_attempts):
            target = np.array(
                [
                    rng.uniform(*self.sim_config.target_x_range),
                    rng.uniform(*self.sim_config.target_y_range),
                    rng.uniform(*self.sim_config.target_z_range),
                ],
                dtype=float,
            )
            if not self._inside_workspace(target):
                continue
            if any(
                np.linalg.norm(target - obstacle.center)
                < obstacle.radius + self.target_radius + self.sim_config.min_target_gap
                for obstacle in obstacles
            ):
                continue
            target_rotvec = rng.uniform(
                self.sim_config.target_rotvec_range[0],
                self.sim_config.target_rotvec_range[1],
                size=3,
            ).astype(float)
            return target, target_rotvec
        raise RuntimeError("Failed to sample a reachable 3D target.")

    def sample_scenario(self, rng: np.random.Generator) -> Scenario3D:
        obstacles = self._sample_obstacles(rng)
        target, target_rotvec = self._sample_target_pose(rng, obstacles)
        tau = np.zeros(
            (
                self.model_config.robot_params.num_segments,
                self.model_config.robot_params.tendons_per_segment,
            ),
            dtype=float,
        )
        return Scenario3D(obstacles=obstacles, target=target, target_rotvec=target_rotvec, tau=tau)

    def _validate_obstacles(self, obstacles: list[Obstacle3D]) -> None:
        if len(obstacles) > self.sim_config.obstacle_count:
            raise ValueError(
                f"Received {len(obstacles)} obstacles, but this simulator is configured for "
                f"at most {self.sim_config.obstacle_count} obstacles."
            )
        for index, obstacle in enumerate(obstacles):
            if np.linalg.norm(obstacle.center) < self.sim_config.base_clearance + obstacle.radius:
                raise ValueError(f"3D obstacle {index} is too close to the base.")
            for other_index in range(index + 1, len(obstacles)):
                other = obstacles[other_index]
                if np.linalg.norm(obstacle.center - other.center) < (
                    obstacle.radius + other.radius + self.sim_config.min_obstacle_gap
                ):
                    raise ValueError(f"3D obstacle {index} overlaps obstacle {other_index}.")

    def _validate_target(self, target: np.ndarray, target_rotvec: np.ndarray, obstacles: list[Obstacle3D]) -> None:
        if not self._inside_workspace(target):
            raise ValueError("3D target is outside the reachable workspace.")
        if not np.isfinite(target_rotvec).all():
            raise ValueError("3D target_rotvec must be finite.")
        for index, obstacle in enumerate(obstacles):
            if np.linalg.norm(target - obstacle.center) < obstacle.radius + self.target_radius + self.sim_config.min_target_gap:
                raise ValueError(f"3D target overlaps obstacle {index}.")

    def _resolve_scenario(
        self,
        rng: np.random.Generator,
        scenario: Scenario3D | dict | str | None,
    ) -> Scenario3D:
        base = self.sample_scenario(rng)
        if scenario is None:
            return base
        if isinstance(scenario, str):
            scenario = get_preset_scenario(scenario)
        if isinstance(scenario, dict):
            if "scenario_name" in scenario:
                base = get_preset_scenario(str(scenario["scenario_name"]))
            scenario = Scenario3D(**{key: value for key, value in scenario.items() if key != "scenario_name"})
        if not isinstance(scenario, Scenario3D):
            raise TypeError("scenario must be None, a str, a dict, or a Scenario3D instance.")

        obstacles = base.obstacles if scenario.obstacles is None else ensure_obstacles(scenario.obstacles)
        target = base.target if scenario.target is None else np.asarray(scenario.target, dtype=float).reshape(3).copy()
        target_rotvec = (
            base.target_rotvec
            if scenario.target_rotvec is None
            else np.asarray(scenario.target_rotvec, dtype=float).reshape(3).copy()
        )
        tau = (
            np.zeros(
                (
                    self.model_config.robot_params.num_segments,
                    self.model_config.robot_params.tendons_per_segment,
                ),
                dtype=float,
            )
            if scenario.tau is None
            else np.asarray(scenario.tau, dtype=float).copy()
        )

        self._validate_obstacles(obstacles)
        self._validate_target(target, target_rotvec, obstacles)
        return Scenario3D(
            name=base.name if scenario.name is None else scenario.name,
            obstacles=obstacles,
            target=target,
            target_rotvec=target_rotvec,
            tau=tau,
        )

    def reset(self, rng: np.random.Generator, scenario: Scenario3D | dict | str | None = None) -> Scenario3D:
        resolved = self._resolve_scenario(rng, scenario)
        self.obstacles = ensure_obstacles(resolved.obstacles or [])
        self.target = np.asarray(resolved.target, dtype=float).reshape(3).copy()
        self.target_rotvec = np.asarray(resolved.target_rotvec, dtype=float).reshape(3).copy()
        tau = np.asarray(resolved.tau, dtype=float).copy()
        self.model.reset(obstacles=self.obstacles, tau=tau)
        self.current_scenario = Scenario3D(
            name=resolved.name,
            obstacles=self.obstacles,
            target=self.target.copy(),
            target_rotvec=self.target_rotvec.copy(),
            tau=self.model.tau.copy(),
        )
        self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace = None
        return self.current_scenario

    def snapshot(self) -> dict[str, object]:
        return {
            "obstacles": ensure_obstacles(self.obstacles),
            "target": self.target.copy(),
            "target_rotvec": self.target_rotvec.copy(),
            "current_scenario": clone_scenario(self.current_scenario),
            "model_state": self.model.snapshot(),
            "control_contact_summary": self._control_contact_summary.copy(),
        }

    def restore(self, state: dict[str, object]) -> None:
        self.obstacles = ensure_obstacles(state.get("obstacles", []))
        self.target = np.asarray(state["target"], dtype=float).reshape(3).copy()
        self.target_rotvec = np.asarray(state["target_rotvec"], dtype=float).reshape(3).copy()
        self.current_scenario = clone_scenario(state.get("current_scenario", Scenario3D()))
        self.model.restore(state["model_state"])
        self.current_scenario.obstacles = ensure_obstacles(self.obstacles)
        self.current_scenario.target = self.target.copy()
        self.current_scenario.target_rotvec = self.target_rotvec.copy()
        self.current_scenario.tau = self.model.tau.copy()
        control_contact_summary = state.get("control_contact_summary")
        if isinstance(control_contact_summary, ContactSummary3D):
            self._control_contact_summary = control_contact_summary.copy()
        else:
            self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace = None

    def clone(self) -> "SoftArmSim3D":
        cloned = SoftArmSim3D(model_config=self.model_config, sim_config=self.sim_config)
        cloned.restore(self.snapshot())
        return cloned

    def control_micro_tau_targets(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_tau = np.asarray(tau, dtype=float).copy()
        previous_tau = self.model.tau.copy()
        decimation = max(int(self.decimation), 1)
        # First-order hold across the control period: the target tension is
        # approached linearly over the micro-steps instead of being applied as
        # an instantaneous jump on the first physics solve.
        return previous_tau, (target_tau - previous_tau) / float(decimation)

    def step_micro(self, tau: np.ndarray) -> ContactSummary3D:
        self.model.apply_action(tau=tau)
        return self.model.contact_summary.copy()

    def finalize_control_step(self, micro_contact_summaries: list[ContactSummary3D]) -> ControlStepTrace3D:
        summaries = [summary.copy() for summary in micro_contact_summaries]
        if not summaries:
            summaries = [self.model.contact_summary.copy()]
        state_priority = {"free": 0, "near": 1, "collision": 2}
        reference = min(
            summaries,
            key=lambda summary: (
                float(summary.min_clearance),
                -float(summary.primary_penetration),
            ),
        )
        segment_clearances = np.full(reference.segment_clearances.shape, float("inf"), dtype=float)
        segment_contact_mask = np.zeros(reference.segment_contact_mask.shape, dtype=bool)
        active_count = 0
        collision_count = 0
        max_penetration = 0.0
        max_force = np.zeros_like(reference.contact_force, dtype=float)
        for summary in summaries:
            if bool(summary.active):
                active_count += 1
            collision_count = max(collision_count, int(summary.collision_count))
            max_penetration = max(max_penetration, float(summary.primary_penetration))
            if np.linalg.norm(summary.contact_force) > np.linalg.norm(max_force):
                max_force = np.asarray(summary.contact_force, dtype=float).copy()
            segment_clearances = np.minimum(segment_clearances, np.asarray(summary.segment_clearances, dtype=float))
            segment_contact_mask |= np.asarray(summary.segment_contact_mask, dtype=bool)
        aggregated_summary = ContactSummary3D(
            active=bool(active_count > 0),
            contact_state=max(
                summaries,
                key=lambda summary: state_priority.get(str(summary.contact_state), 0),
            ).contact_state,
            collision_count=int(collision_count),
            min_clearance=float(reference.min_clearance),
            nearest_obstacle_index=int(reference.nearest_obstacle_index),
            nearest_sample_index=int(reference.nearest_sample_index),
            nearest_segment_index=int(reference.nearest_segment_index),
            nearest_s=float(reference.nearest_s),
            nearest_point=np.asarray(reference.nearest_point, dtype=float).copy(),
            nearest_obstacle_center=np.asarray(reference.nearest_obstacle_center, dtype=float).copy(),
            nearest_obstacle_radii=np.asarray(reference.nearest_obstacle_radii, dtype=float).copy(),
            contact_normal=np.asarray(reference.contact_normal, dtype=float).copy(),
            contact_force=max_force,
            primary_penetration=float(max_penetration),
            segment_clearances=segment_clearances,
            segment_contact_mask=segment_contact_mask,
            contact_occurrence_rate=float(active_count) / float(len(summaries)),
            micro_step_count=int(len(summaries)),
        )
        self._control_contact_summary = aggregated_summary.copy()
        self.current_scenario.tau = self.model.tau.copy()
        trace = ControlStepTrace3D(
            sim_dt=float(self.sim_config.sim_dt),
            control_dt=float(self.sim_config.control_dt),
            decimation=int(self.decimation),
            micro_contact_summaries=tuple(summary.copy() for summary in summaries),
            contact_summary=aggregated_summary.copy(),
        )
        self._last_control_trace = trace
        return trace

    def step(self, tau: np.ndarray) -> ControlStepTrace3D:
        previous_tau, tau_step = self.control_micro_tau_targets(tau)
        micro_contact_summaries: list[ContactSummary3D] = []
        current_tau = previous_tau.copy()
        for _ in range(self.decimation):
            current_tau = current_tau + tau_step
            summary = self.step_micro(current_tau)
            micro_contact_summaries.append(summary)
        return self.finalize_control_step(micro_contact_summaries)

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.model.tip_position - self.target))

    def orientation_error_to_target(self) -> float:
        return float(np.linalg.norm(self.target_rotvec - self.model.tip_rotvec))

    @property
    def contact_summary(self):
        return self._control_contact_summary

    def scene_state(self) -> dict:
        return {
            "base_position": self.model.root_position.copy(),
            "tip_position": self.model.tip_position.copy(),
            "tip_rotvec": self.model.tip_rotvec.copy(),
            "free_points": self.model.free_polyline(),
            "observed_points": self.model.observed_polyline(),
            "backbone_points": self.model.backbone_points(6),
            "obstacles": self.obstacles,
            "target": self.target.copy(),
            "target_rotvec": self.target_rotvec.copy(),
            "target_radius": self.target_radius,
            "target_orientation_tolerance": float(self.target_orientation_tolerance),
            "collision_points": [point.copy() for point in self.model.collision_events],
            "contact_summary": self.contact_summary.as_dict(),
            "tau": self.model.tau.copy(),
        }


__all__ = [
    "PRESET_SCENARIOS",
    "Scenario3D",
    "SoftArmSim3D",
    "SoftArmSim3DConfig",
    "clone_scenario",
    "get_preset_scenario",
    "list_preset_scenarios",
]
