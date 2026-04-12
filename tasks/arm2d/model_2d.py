from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.pcc import ContinuumArm2D, PCCSegment2D, clone_segments
from common.utils import wrap_angle

from .physics_2d import (
    CircleObstacle,
    CollisionEvent,
    apply_collision_response,
    ensure_obstacles,
    find_first_collision,
)


@dataclass(frozen=True)
class ArmModel2DConfig:
    segment_lengths: tuple[float, ...] = (12.0, 14.0)
    base_x_limits: tuple[float, float] = (0.0, 12.0)
    curvature_limit: float = 0.18
    arm_radius: float = 1.2
    max_collision_responses: int = 6
    min_segment_length: float = 1.0
    render_points_per_segment: int = 32


class ArmModel2D:
    def __init__(
        self,
        config: ArmModel2DConfig | None = None,
        obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None,
        base_x: float = 0.0,
        kappas: np.ndarray | None = None,
    ):
        self.config = config or ArmModel2DConfig()
        self.base_x = float(np.clip(base_x, *self.config.base_x_limits))
        self.free_kappas = self._sanitize_kappas(kappas)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.free_segments: list[PCCSegment2D] = []
        self.observed_segments: list[PCCSegment2D] = []
        self.free_chain = ContinuumArm2D([])
        self.observed_chain = ContinuumArm2D([])
        self.collision_events: list[CollisionEvent] = []
        self.rebuild()

    def _sanitize_kappas(self, kappas: np.ndarray | None) -> np.ndarray:
        if kappas is None:
            values = np.zeros(len(self.config.segment_lengths), dtype=float)
        else:
            values = np.asarray(kappas, dtype=float).copy()
        if values.shape != (len(self.config.segment_lengths),):
            raise ValueError("kappas must match the number of controllable segments.")
        return np.clip(values, -self.config.curvature_limit, self.config.curvature_limit)

    @property
    def root_position(self) -> np.ndarray:
        return np.array([self.base_x, 0.0], dtype=float)

    @property
    def root_direction(self) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=float)

    def _build_free_segments(self) -> list[PCCSegment2D]:
        return [
            PCCSegment2D(length=length, kappa=kappa)
            for length, kappa in zip(self.config.segment_lengths, self.free_kappas)
        ]

    def rebuild(self) -> None:
        self.free_segments = self._build_free_segments()
        self.free_chain = ContinuumArm2D(self.free_segments)
        self.free_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        observed_segments = clone_segments(self.free_segments)
        observed_chain = ContinuumArm2D(observed_segments)
        observed_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        collision_events: list[CollisionEvent] = []
        for _ in range(self.config.max_collision_responses):
            event = find_first_collision(
                segments=observed_chain.segments,
                obstacles=self.obstacles,
                arm_radius=self.config.arm_radius,
            )
            if event is None:
                break

            collision_events.append(event)
            observed_segments = apply_collision_response(
                segments=observed_chain.segments,
                event=event,
                obstacle=self.obstacles[event.obstacle_index],
                arm_radius=self.config.arm_radius,
                min_segment_length=self.config.min_segment_length,
            )
            observed_chain = ContinuumArm2D(observed_segments)
            observed_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        self.observed_segments = observed_chain.segments
        self.observed_chain = observed_chain
        self.collision_events = collision_events

    def reset(
        self,
        *,
        obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None,
        base_x: float = 0.0,
        kappas: np.ndarray | None = None,
    ) -> None:
        self.base_x = float(np.clip(base_x, *self.config.base_x_limits))
        self.free_kappas = self._sanitize_kappas(kappas)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.rebuild()

    def set_obstacles(self, obstacles: list[CircleObstacle | tuple[float, float, float]]) -> None:
        self.obstacles = ensure_obstacles(obstacles)
        self.rebuild()

    def apply_action(self, base_delta: float, kappa_delta: np.ndarray) -> None:
        kappa_delta = np.asarray(kappa_delta, dtype=float)
        if kappa_delta.shape != self.free_kappas.shape:
            raise ValueError("kappa_delta must match the number of controllable segments.")
        self.base_x = float(np.clip(self.base_x + base_delta, *self.config.base_x_limits))
        self.free_kappas = np.clip(
            self.free_kappas + kappa_delta,
            -self.config.curvature_limit,
            self.config.curvature_limit,
        )
        self.rebuild()

    @property
    def tip_position(self) -> np.ndarray:
        return self.observed_chain.tip_position

    @property
    def tip_angle(self) -> float:
        return wrap_angle(self.observed_chain.tip_angle)

    def tip_state(self) -> np.ndarray:
        return np.array(
            [self.tip_position[0], self.tip_position[1], self.tip_angle],
            dtype=np.float32,
        )

    def free_polyline(self) -> np.ndarray:
        return self.free_chain.sample_points(self.config.render_points_per_segment)

    def observed_polyline(self) -> np.ndarray:
        return self.observed_chain.sample_points(self.config.render_points_per_segment)
