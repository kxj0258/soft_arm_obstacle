from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_2d import ArmModel2D, ArmModel2DConfig
from .physics_2d import CircleObstacle


@dataclass(frozen=True)
class SoftArmSim2DConfig:
    obstacle_count: int = 3
    obstacle_radius_range: tuple[float, float] = (2.2, 4.0)
    obstacle_x_range: tuple[float, float] = (10.0, 28.0)
    obstacle_y_range: tuple[float, float] = (-12.0, 12.0)
    target_x_range: tuple[float, float] = (18.0, 36.0)
    target_y_range: tuple[float, float] = (-14.0, 14.0)
    target_radius: float = 2.5
    min_obstacle_gap: float = 2.5
    min_target_gap: float = 2.5
    base_clearance: float = 8.0
    max_sampling_attempts: int = 256


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

    def reset(self, rng: np.random.Generator) -> None:
        self.obstacles = self._sample_obstacles(rng)
        self.target = self._sample_target(rng, self.obstacles)
        self.model.reset(obstacles=self.obstacles, base_x=0.0, kappas=np.zeros(len(self.model_config.segment_lengths)))

    def step(self, base_delta: float, kappa_delta: np.ndarray) -> None:
        self.model.apply_action(base_delta=base_delta, kappa_delta=kappa_delta)

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.model.tip_position - self.target))

    def scene_state(self) -> dict:
        return {
            "base_position": self.model.root_position.copy(),
            "tip_position": self.model.tip_position.copy(),
            "tip_angle": self.model.tip_angle,
            "free_points": self.model.free_polyline(),
            "observed_points": self.model.observed_polyline(),
            "obstacles": self.obstacles,
            "target": self.target.copy(),
            "target_radius": self.target_radius,
            "collision_points": [event.contact_point.copy() for event in self.model.collision_events],
        }
