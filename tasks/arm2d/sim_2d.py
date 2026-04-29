from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_2d import ArmModel2D, ArmModel2DConfig
from .physics_2d import CircleObstacle, ensure_obstacles


@dataclass(frozen=True)
class SoftArmSim2DConfig:
    obstacle_count: int = 6
    obstacle_radius_range: tuple[float, float] = (2.2, 4.0)
    obstacle_x_range: tuple[float, float] = (10.0, 28.0)
    obstacle_y_range: tuple[float, float] = (-12.0, 12.0)
    target_x_range: tuple[float, float] = (18.0, 36.0)
    target_y_range: tuple[float, float] = (-14.0, 14.0)
    target_radius: float = 0.25
    min_obstacle_gap: float = 1.0
    min_target_gap: float = 2.5
    base_clearance: float = 8.0
    max_sampling_attempts: int = 256


@dataclass
class Scenario2D:
    name: str | None = None
    obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None
    target: np.ndarray | tuple[float, float] | None = None
    base_x: float | None = None
    kappas: np.ndarray | None = None
    tendon_lengths: np.ndarray | None = None


PRESET_SCENARIOS: dict[str, Scenario2D] = {
    "reach_easy": Scenario2D(
        name="reach_easy",
        obstacles=[(14.0, 6.5, 2.7)],
        target=(24.0, 2.0),
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
    "contact_left": Scenario2D(
        name="contact_left",
        obstacles=[(10.0, 5.0, 1.0), (12.5, 1.0, 1.0), (15.0, -4.0, 1.0), (17.5, 1.0, 1.0), (20.0, -2.0, 1.0), (20.0, 5.0, 1.0)],
        target=(25.0, 5.0),
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
    "contact_channel": Scenario2D(
        name="contact_channel",
        obstacles=[(11.0, 5.5, 2.8), (17.5, -4.8, 3.0), (24.0, 4.8, 2.5)],
        target=(31.0, 0.5),
        base_x=0.0,
        kappas=np.zeros(2, dtype=float),
    ),
}


def clone_scenario(scenario: Scenario2D) -> Scenario2D:
    return Scenario2D(
        name=scenario.name,
        obstacles=None if scenario.obstacles is None else ensure_obstacles(scenario.obstacles),
        target=None if scenario.target is None else np.asarray(scenario.target, dtype=float).copy(),
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
        self.target_radius = self.sim_config.target_radius
        self.current_scenario = Scenario2D()

    @property
    def max_reach(self) -> float:
        return self.model_config.base_x_limits[1] + sum(self.model_config.segment_lengths)

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

    def _sample_target(
        self,
        rng: np.random.Generator,
        obstacles: list[CircleObstacle],
    ) -> np.ndarray:
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
            return target
        raise RuntimeError("Failed to sample a reachable target.")

    def sample_scenario(self, rng: np.random.Generator) -> Scenario2D:
        obstacles = self._sample_obstacles(rng)
        target = self._sample_target(rng, obstacles)
        kappas = np.zeros(len(self.model_config.segment_lengths), dtype=float)
        return Scenario2D(obstacles=obstacles, target=target, base_x=0.0, kappas=kappas, tendon_lengths=None)

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

    def _validate_target(self, target: np.ndarray, obstacles: list[CircleObstacle]) -> None:
        if np.linalg.norm(target) > self.max_reach - self.target_radius:
            raise ValueError("Target is outside the reachable workspace.")
        for index, obstacle in enumerate(obstacles):
            if np.linalg.norm(target - obstacle.center) < obstacle.radius + self.target_radius + self.sim_config.min_target_gap:
                raise ValueError(f"Target overlaps obstacle {index}.")

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
        self._validate_target(target, obstacles)
        return Scenario2D(
            name=base.name if scenario.name is None else scenario.name,
            obstacles=obstacles,
            target=target,
            base_x=base_x,
            kappas=kappas,
            tendon_lengths=tendon_lengths,
        )

    def reset(self, rng: np.random.Generator, scenario: Scenario2D | dict | str | None = None) -> Scenario2D:
        resolved = self._resolve_scenario(rng, scenario)
        self.obstacles = ensure_obstacles(resolved.obstacles or [])
        self.target = np.asarray(resolved.target, dtype=float).copy()
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
            base_x=self.model.base_x,
            kappas=self.model.free_kappas.copy(),
            tendon_lengths=self.model.free_tendon_lengths.copy(),
        )
        return self.current_scenario

    def step(self, base_delta: float, tendon_length_deltas: np.ndarray) -> None:
        self.model.apply_action(base_delta=base_delta, tendon_length_deltas=tendon_length_deltas)

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.model.tip_position - self.target))

    @property
    def contact_summary(self):
        return self.model.contact_summary

    def scene_state(self) -> dict:
        return {
            "base_position": self.model.root_position.copy(),
            "tip_position": self.model.tip_position.copy(),
            "tip_angle": self.model.tip_angle,
            "free_points": self.model.free_polyline(),
            "observed_points": self.model.observed_polyline(),
            "backbone_points": self.model.backbone_points(4),
            "obstacles": self.obstacles,
            "target": self.target.copy(),
            "target_radius": self.target_radius,
            "collision_points": [event.contact_point.copy() for event in self.model.collision_events],
            "contact_summary": self.model.contact_summary.as_dict(),
        }
