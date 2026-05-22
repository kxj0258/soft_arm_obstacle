from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import normalize
from common.utils import wrap_angle

from .model_2d import ArmModel2D, ArmModel2DConfig
from .physics_2d import ContactSummary, CircleObstacle, aggregate_contact_summaries, ensure_obstacles


@dataclass(frozen=True)
class SoftArmSim2DConfig:
    obstacle_count: int = 6
    obstacle_radius_range: tuple[float, float] = (2.2, 4.0)
    obstacle_x_range: tuple[float, float] = (10.0, 28.0)
    obstacle_y_range: tuple[float, float] = (-12.0, 12.0)
    target_x_range: tuple[float, float] = (18.0, 36.0)
    target_y_range: tuple[float, float] = (-14.0, 14.0)
    target_angle_range: tuple[float, float] = (-float(np.pi), float(np.pi))
    target_radius: float = 0.25
    target_angle_tolerance: float = 0.25
    min_obstacle_gap: float = 1.0
    min_target_gap: float = 2.5
    base_clearance: float = 8.0
    max_sampling_attempts: int = 256
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
class ControlStepTrace2D:
    sim_dt: float
    control_dt: float
    decimation: int
    micro_contact_summaries: tuple[ContactSummary, ...]
    contact_summary: ContactSummary
    action_blocked: bool


@dataclass
class Scenario2D:
    name: str | None = None
    obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None
    target: np.ndarray | tuple[float, float] | None = None
    target_angle: float | None = None
    base_x: float | None = None
    kappas: np.ndarray | None = None
    tendon_lengths: np.ndarray | None = None


PRESET_SCENARIOS: dict[str, Scenario2D] = {
    "reach_easy": Scenario2D(
        name="reach_easy",
        obstacles=[(14.0, 6.5, 2.7)],
        target=(24.0, 2.0),
        target_angle=0.0,
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
    "contact_left": Scenario2D(
        name="contact_left",
        obstacles=[(10.0, 5.0, 1.0), (12.5, 7.5, 1.5), (15.0, -5.0, 1.5), (17.5, 5.0, 1.5), (20.0, -3.0, 1.5)],
        target=(25.0, 7.0),
        target_angle=0.0,
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
    "contact_channel": Scenario2D(
        name="contact_channel",
        obstacles=[(11.0, 5.5, 2.8), (17.5, -4.8, 3.0), (24.0, 4.8, 2.5)],
        target=(31.0, 0.5),
        target_angle=0.0,
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
}


def clone_scenario(scenario: Scenario2D) -> Scenario2D:
    return Scenario2D(
        name=scenario.name,
        obstacles=None if scenario.obstacles is None else ensure_obstacles(scenario.obstacles),
        target=None if scenario.target is None else np.asarray(scenario.target, dtype=float).copy(),
        target_angle=None if scenario.target_angle is None else float(scenario.target_angle),
        base_x=scenario.base_x,
        kappas=None if scenario.kappas is None else np.asarray(scenario.kappas, dtype=float).copy(),
        tendon_lengths=None if scenario.tendon_lengths is None else np.asarray(scenario.tendon_lengths, dtype=float).copy(),
    )


def list_preset_scenarios() -> tuple[str, ...]:
    return tuple(PRESET_SCENARIOS.keys())


def get_preset_scenario(name: str) -> Scenario2D:
    if name not in PRESET_SCENARIOS:
        available = ", ".join(list_preset_scenarios())
        raise KeyError(f"Unknown preset scenario {name!r}. Available presets: {available}")
    return clone_scenario(PRESET_SCENARIOS[name])


class SoftArmSim2D:
    def __init__(
        self,
        model_config: ArmModel2DConfig | None = None,
        sim_config: SoftArmSim2DConfig | None = None,
    ):
        self.model_config = model_config or ArmModel2DConfig()
        self.sim_config = sim_config or SoftArmSim2DConfig()
        self.model = ArmModel2D(config=self.model_config)
        self.obstacles: list[CircleObstacle] = []
        self.target = np.zeros(2, dtype=float)
        self.target_angle = 0.0
        self.target_radius = self.sim_config.target_radius
        self.target_angle_tolerance = self.sim_config.target_angle_tolerance
        self.current_scenario = Scenario2D()
        self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace: ControlStepTrace2D | None = None

    @property
    def max_reach(self) -> float:
        return self.model_config.base_x_limits[1] + sum(self.model_config.segment_lengths)

    @property
    def decimation(self) -> int:
        return self.sim_config.decimation

    def _sample_obstacles(self, rng: np.random.Generator) -> list[CircleObstacle]:
        obstacles: list[CircleObstacle] = []
        for _ in range(self.sim_config.obstacle_count):
            for _ in range(self.sim_config.max_sampling_attempts):
                radius = rng.uniform(*self.sim_config.obstacle_radius_range)
                center = np.array(
                    [
                        rng.uniform(*self.sim_config.obstacle_x_range),
                        rng.uniform(*self.sim_config.obstacle_y_range),
                    ],
                    dtype=float,
                )
                if np.linalg.norm(center) < self.sim_config.base_clearance + radius:
                    continue
                if any(
                    np.linalg.norm(center - obstacle.center)
                    < (radius + obstacle.radius + self.sim_config.min_obstacle_gap)
                    for obstacle in obstacles
                ):
                    continue
                obstacles.append(CircleObstacle(center=center, radius=radius))
                break
            else:
                raise RuntimeError("Failed to sample a collision-free obstacle layout.")
        return obstacles

    def _sample_target_pose(
        self,
        rng: np.random.Generator,
        obstacles: list[CircleObstacle],
    ) -> tuple[np.ndarray, float]:
        for _ in range(self.sim_config.max_sampling_attempts):
            target = np.array(
                [
                    rng.uniform(*self.sim_config.target_x_range),
                    rng.uniform(*self.sim_config.target_y_range),
                ],
                dtype=float,
            )
            if np.linalg.norm(target) > self.max_reach - self.sim_config.target_radius:
                continue
            if any(
                np.linalg.norm(target - obstacle.center)
                < obstacle.radius + self.target_radius + self.sim_config.min_target_gap
                for obstacle in obstacles
            ):
                continue
            target_angle = wrap_angle(rng.uniform(*self.sim_config.target_angle_range))
            return target, target_angle
        raise RuntimeError("Failed to sample a reachable target.")

    def sample_scenario(self, rng: np.random.Generator) -> Scenario2D:
        obstacles = self._sample_obstacles(rng)
        target, target_angle = self._sample_target_pose(rng, obstacles)
        kappas = np.zeros(len(self.model_config.segment_lengths), dtype=float)
        return Scenario2D(
            obstacles=obstacles,
            target=target,
            target_angle=target_angle,
            base_x=0.0,
            kappas=kappas,
            tendon_lengths=None,
        )

    def _validate_obstacles(self, obstacles: list[CircleObstacle]) -> None:
        if len(obstacles) > self.sim_config.obstacle_count:
            raise ValueError(
                f"Received {len(obstacles)} obstacles, but this simulator is configured for "
                f"at most {self.sim_config.obstacle_count} obstacles."
            )
        for index, obstacle in enumerate(obstacles):
            if np.linalg.norm(obstacle.center) < self.sim_config.base_clearance + obstacle.radius:
                raise ValueError(f"Obstacle {index} is too close to the base.")
            for other_index in range(index + 1, len(obstacles)):
                other = obstacles[other_index]
                if np.linalg.norm(obstacle.center - other.center) < (
                    obstacle.radius + other.radius + self.sim_config.min_obstacle_gap
                ):
                    raise ValueError(f"Obstacle {index} overlaps obstacle {other_index}.")

    def _validate_target(self, target: np.ndarray, target_angle: float, obstacles: list[CircleObstacle]) -> None:
        if np.linalg.norm(target) > self.max_reach - self.target_radius:
            raise ValueError("Target is outside the reachable workspace.")
        for index, obstacle in enumerate(obstacles):
            if np.linalg.norm(target - obstacle.center) < obstacle.radius + self.target_radius + self.sim_config.min_target_gap:
                raise ValueError(f"Target overlaps obstacle {index}.")
        if not np.isfinite(float(target_angle)):
            raise ValueError("Target angle must be finite.")

    def _resolve_scenario(
        self,
        rng: np.random.Generator,
        scenario: Scenario2D | dict | str | None,
    ) -> Scenario2D:
        base = self.sample_scenario(rng)
        if scenario is None:
            return base
        if isinstance(scenario, str):
            scenario = get_preset_scenario(scenario)
        if isinstance(scenario, dict):
            if "scenario_name" in scenario:
                base = get_preset_scenario(str(scenario["scenario_name"]))
            scenario = Scenario2D(**{key: value for key, value in scenario.items() if key != "scenario_name"})
        if not isinstance(scenario, Scenario2D):
            raise TypeError("scenario must be None, a str, a dict, or a Scenario2D instance.")

        obstacles = base.obstacles if scenario.obstacles is None else ensure_obstacles(scenario.obstacles)
        target = base.target if scenario.target is None else np.asarray(scenario.target, dtype=float).copy()
        target_angle = base.target_angle if scenario.target_angle is None else wrap_angle(float(scenario.target_angle))
        base_x = 0.0 if scenario.base_x is None else float(scenario.base_x)
        kappas = (
            np.zeros(len(self.model_config.segment_lengths), dtype=float)
            if scenario.kappas is None
            else np.asarray(scenario.kappas, dtype=float).copy()
        )
        tendon_lengths = (
            None
            if scenario.tendon_lengths is None
            else np.asarray(scenario.tendon_lengths, dtype=float).copy()
        )

        self._validate_obstacles(obstacles)
        self._validate_target(target, target_angle, obstacles)
        return Scenario2D(
            name=base.name if scenario.name is None else scenario.name,
            obstacles=obstacles,
            target=target,
            target_angle=target_angle,
            base_x=base_x,
            kappas=kappas,
            tendon_lengths=tendon_lengths,
        )

    def reset(self, rng: np.random.Generator, scenario: Scenario2D | dict | str | None = None) -> Scenario2D:
        resolved = self._resolve_scenario(rng, scenario)
        self.obstacles = ensure_obstacles(resolved.obstacles or [])
        self.target = np.asarray(resolved.target, dtype=float).copy()
        self.target_angle = 0.0 if resolved.target_angle is None else wrap_angle(float(resolved.target_angle))
        kappas = (
            np.zeros(len(self.model_config.segment_lengths), dtype=float)
            if resolved.kappas is None
            else np.asarray(resolved.kappas, dtype=float).copy()
        )
        tendon_lengths = (
            None
            if resolved.tendon_lengths is None
            else np.asarray(resolved.tendon_lengths, dtype=float).copy()
        )
        base_x = 0.0 if resolved.base_x is None else float(resolved.base_x)
        self.model.reset(
            obstacles=self.obstacles,
            base_x=base_x,
            kappas=kappas,
            tendon_lengths=tendon_lengths,
        )
        self.current_scenario = Scenario2D(
            name=resolved.name,
            obstacles=self.obstacles,
            target=self.target.copy(),
            target_angle=self.target_angle,
            base_x=self.model.base_x,
            kappas=self.model.free_kappas.copy(),
            tendon_lengths=self.model.free_tendon_lengths.copy(),
        )
        self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace = None
        return self.current_scenario

    def snapshot(self) -> dict[str, object]:
        return {
            "obstacles": ensure_obstacles(self.obstacles),
            "target": self.target.copy(),
            "target_angle": float(self.target_angle),
            "current_scenario": clone_scenario(self.current_scenario),
            "model_state": self.model.snapshot(),
            "control_contact_summary": self._control_contact_summary.copy(),
        }

    def restore(self, state: dict[str, object]) -> None:
        self.obstacles = ensure_obstacles(state.get("obstacles", []))
        self.target = np.asarray(state["target"], dtype=float).copy()
        self.target_angle = wrap_angle(float(state["target_angle"]))
        self.current_scenario = clone_scenario(state.get("current_scenario", Scenario2D()))
        self.model.restore(state["model_state"])
        self.current_scenario.obstacles = ensure_obstacles(self.obstacles)
        self.current_scenario.target = self.target.copy()
        self.current_scenario.target_angle = float(self.target_angle)
        self.current_scenario.base_x = float(self.model.base_x)
        self.current_scenario.kappas = self.model.free_kappas.copy()
        self.current_scenario.tendon_lengths = self.model.free_tendon_lengths.copy()
        control_contact_summary = state.get("control_contact_summary")
        if isinstance(control_contact_summary, ContactSummary):
            self._control_contact_summary = control_contact_summary.copy()
        else:
            self._control_contact_summary = self.model.contact_summary.copy()
        self._last_control_trace = None

    def clone(self) -> "SoftArmSim2D":
        cloned = SoftArmSim2D(model_config=self.model_config, sim_config=self.sim_config)
        cloned.restore(self.snapshot())
        return cloned

    def control_micro_deltas(self, *, base_delta: float, tendon_length_deltas: np.ndarray) -> tuple[float, np.ndarray]:
        tendon_length_deltas = np.asarray(tendon_length_deltas, dtype=float)
        if tendon_length_deltas.shape != self.model.free_tendon_lengths.shape:
            raise ValueError("tendon_length_deltas must have shape (num_segments, 2).")
        decimation = max(int(self.decimation), 1)
        # A low-rate control increment is distributed evenly over the high-rate
        # simulator ticks so the actuator motion is first-order held instead of
        # being injected as an impulse on the first physics update.
        return float(base_delta) / float(decimation), tendon_length_deltas / float(decimation)

    def step_micro(self, *, base_delta: float, tendon_length_deltas: np.ndarray) -> ContactSummary:
        self.model.apply_action(base_delta=base_delta, tendon_length_deltas=tendon_length_deltas)
        return self.model.contact_summary.copy()

    def finalize_control_step(
        self,
        micro_contact_summaries: list[ContactSummary],
        *,
        action_blocked: bool,
    ) -> ControlStepTrace2D:
        summaries = [summary.copy() for summary in micro_contact_summaries]
        if not summaries:
            summaries = [self.model.contact_summary.copy()]
        aggregated_summary = aggregate_contact_summaries(summaries)
        self._control_contact_summary = aggregated_summary.copy()
        self.current_scenario.base_x = float(self.model.base_x)
        self.current_scenario.kappas = self.model.free_kappas.copy()
        self.current_scenario.tendon_lengths = self.model.free_tendon_lengths.copy()
        trace = ControlStepTrace2D(
            sim_dt=float(self.sim_config.sim_dt),
            control_dt=float(self.sim_config.control_dt),
            decimation=int(self.decimation),
            micro_contact_summaries=tuple(summary.copy() for summary in summaries),
            contact_summary=aggregated_summary.copy(),
            action_blocked=bool(action_blocked),
        )
        self._last_control_trace = trace
        return trace

    def step(self, base_delta: float, tendon_length_deltas: np.ndarray) -> ControlStepTrace2D:
        micro_base_delta, micro_tendon_deltas = self.control_micro_deltas(
            base_delta=base_delta,
            tendon_length_deltas=tendon_length_deltas,
        )
        micro_contact_summaries: list[ContactSummary] = []
        action_blocked = False
        for micro_index in range(self.decimation):
            summary = self.step_micro(
                base_delta=micro_base_delta,
                tendon_length_deltas=micro_tendon_deltas,
            )
            micro_contact_summaries.append(summary)
            if self.model.action_blocked:
                action_blocked = True
                frozen_summary = self.model.contact_summary.copy()
                for _ in range(micro_index + 1, self.decimation):
                    micro_contact_summaries.append(frozen_summary.copy())
                break
        return self.finalize_control_step(
            micro_contact_summaries,
            action_blocked=action_blocked,
        )

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.model.tip_position - self.target))

    def angle_error_to_target(self) -> float:
        return wrap_angle(self.target_angle - self.model.tip_angle)

    @property
    def contact_summary(self):
        return self._control_contact_summary

    def scene_state(self) -> dict:
        contact_markers: list[dict[str, np.ndarray | float | str]] = []
        for event in self.model.collision_events:
            contact_markers.append(
                {
                    "point": event.contact_point.copy(),
                    "normal": event.normal.copy(),
                    "magnitude": float(max(event.penetration, 0.0)),
                    "state": "collision",
                }
            )

        if not contact_markers and self.model.contact_summary.active:
            summary = self.model.contact_summary
            normal = normalize(summary.nearest_arm_point - summary.nearest_obstacle_point)
            contact_point = 0.5 * (summary.nearest_arm_point + summary.nearest_obstacle_point)
            contact_markers.append(
                {
                    "point": np.asarray(contact_point, dtype=float),
                    "normal": np.asarray(normal, dtype=float),
                    "magnitude": float(max(summary.primary_penetration, 0.0)),
                    "state": str(summary.contact_state),
                }
            )

        return {
            "base_position": self.model.root_position.copy(),
            "tip_position": self.model.tip_position.copy(),
            "tip_angle": self.model.tip_angle,
            "free_points": self.model.free_polyline(),
            "observed_points": self.model.observed_polyline(),
            "backbone_points": self.model.backbone_points(4),
            "obstacles": self.obstacles,
            "target": self.target.copy(),
            "target_angle": float(self.target_angle),
            "target_radius": self.target_radius,
            "target_angle_tolerance": float(self.target_angle_tolerance),
            "collision_points": [event.contact_point.copy() for event in self.model.collision_events],
            "contact_markers": contact_markers,
            "contact_summary": self.contact_summary.as_dict(),
        }
