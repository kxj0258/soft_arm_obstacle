from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.pcc import ContinuumArm2D, PCCSegment2D, clone_segments
from common.utils import wrap_angle

from .contact_solver_2d import solve_optimization_contact_response
from .physics_2d import (
    CircleObstacle,
    CollisionEvent,
    ContactResponse2DConfig,
    ContactSummary,
    apply_collision_response,
    ensure_obstacles,
    find_first_collision,
    make_empty_contact_summary,
    summarize_contacts,
)


@dataclass(frozen=True)
class ArmModel2DConfig:
    segment_lengths: tuple[float, ...] = (10.0, 10.0)
    base_x_limits: tuple[float, float] = (0.0, 10.0)
    curvature_limit: float = 0.18
    arm_radius: float = 0.2
    tendon_rest_length: float = 10.0
    tendon_routing_radius: float = 0.5
    tendon_length_limits: tuple[float, float] = (5.0, 15.0)
    max_collision_responses: int = 6
    min_segment_length: float = 1.0
    render_points_per_segment: int = 32
    contact_response: ContactResponse2DConfig = field(default_factory=ContactResponse2DConfig)


class ArmModel2D:
    def __init__(
        self,
        config: ArmModel2DConfig | None = None,
        obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None,
        base_x: float = 0.0,
        kappas: np.ndarray | None = None,
        tendon_lengths: np.ndarray | None = None,
    ):
        self.config = config or ArmModel2DConfig()
        self.base_x = float(np.clip(base_x, *self.config.base_x_limits))
        self.free_tendon_lengths = self._resolve_initial_tendon_lengths(
            tendon_lengths=tendon_lengths,
            kappas=kappas,
        )
        self.free_kappas = self._kappas_from_tendon_lengths(self.free_tendon_lengths)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.free_segments: list[PCCSegment2D] = []
        self.observed_segments: list[PCCSegment2D] = []
        self.free_chain = ContinuumArm2D([])
        self.observed_chain = ContinuumArm2D([])
        self.collision_events: list[CollisionEvent] = []
        self.contact_summary = make_empty_contact_summary(len(self.config.segment_lengths))
        self.contact_backend_used = "heuristic"
        self._optimization_warm_start: np.ndarray | None = None
        self._observed_control_slices: tuple[slice, ...] | None = None
        self.rebuild()

    @property
    def num_segments(self) -> int:
        return len(self.config.segment_lengths)

    def _sanitize_kappas(self, kappas: np.ndarray | None) -> np.ndarray:
        if kappas is None:
            values = np.zeros(self.num_segments, dtype=float)
        else:
            values = np.asarray(kappas, dtype=float).copy()
        if values.shape != (self.num_segments,):
            raise ValueError("kappas must match the number of controllable segments.")
        return np.clip(values, -self.config.curvature_limit, self.config.curvature_limit)

    def _default_tendon_lengths(self) -> np.ndarray:
        return np.full((self.num_segments, 2), float(self.config.tendon_rest_length), dtype=float)

    def _sanitize_tendon_lengths(self, tendon_lengths: np.ndarray | None) -> np.ndarray:
        if tendon_lengths is None:
            values = self._default_tendon_lengths()
        else:
            values = np.asarray(tendon_lengths, dtype=float).copy()
            if values.shape == (self.num_segments * 2,):
                values = values.reshape(self.num_segments, 2)
        if values.shape != (self.num_segments, 2):
            raise ValueError("tendon_lengths must have shape (num_segments, 2).")
        lower, upper = self.config.tendon_length_limits
        if upper <= lower:
            raise ValueError("tendon_length_limits must satisfy min < max.")
        return np.clip(values, float(lower), float(upper))

    def _tendon_lengths_from_kappas(self, kappas: np.ndarray) -> np.ndarray:
        kappas = self._sanitize_kappas(kappas)
        lengths = np.asarray(self.config.segment_lengths, dtype=float)
        routing_radius = max(float(self.config.tendon_routing_radius), 1e-8)
        tendon_lengths = self._default_tendon_lengths()
        tendon_lengths[:, 0] -= routing_radius * lengths * kappas
        tendon_lengths[:, 1] += routing_radius * lengths * kappas
        return self._sanitize_tendon_lengths(tendon_lengths)

    def _kappas_from_tendon_lengths(self, tendon_lengths: np.ndarray) -> np.ndarray:
        tendon_lengths = self._sanitize_tendon_lengths(tendon_lengths)
        lengths = np.asarray(self.config.segment_lengths, dtype=float)
        routing_radius = max(float(self.config.tendon_routing_radius), 1e-8)
        differential = tendon_lengths[:, 1] - tendon_lengths[:, 0]
        kappas = differential / (2.0 * routing_radius * np.maximum(lengths, 1e-8))
        return np.clip(kappas, -self.config.curvature_limit, self.config.curvature_limit)

    def _resolve_initial_tendon_lengths(
        self,
        *,
        tendon_lengths: np.ndarray | None,
        kappas: np.ndarray | None,
    ) -> np.ndarray:
        if tendon_lengths is not None:
            return self._sanitize_tendon_lengths(tendon_lengths)
        if kappas is not None:
            return self._tendon_lengths_from_kappas(kappas)
        return self._default_tendon_lengths()

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
        previous_summary = self.contact_summary

        self.free_segments = self._build_free_segments()
        self.free_chain = ContinuumArm2D(self.free_segments)
        self.free_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        response_config = self.config.contact_response
        if response_config.backend == "optimization":
            try:
                solved = solve_optimization_contact_response(
                    segment_lengths=self.config.segment_lengths,
                    segment_kappas=self.free_kappas,
                    obstacles=self.obstacles,
                    arm_radius=self.config.arm_radius,
                    curvature_limit=self.config.curvature_limit,
                    start_position=self.root_position,
                    start_direction=self.root_direction,
                    response_config=response_config,
                    previous_solution=self._optimization_warm_start,
                )
            except (ImportError, RuntimeError):
                if not response_config.fallback_to_heuristic:
                    raise
            else:
                self.observed_segments = solved.segments
                self.observed_chain = solved.chain
                self.collision_events = []
                self._optimization_warm_start = solved.warm_start_kappas.copy()
                self._observed_control_slices = solved.summary_segment_slices
                optimization_summary = summarize_contacts(
                    segments=self.observed_chain.segments,
                    obstacles=self.obstacles,
                    arm_radius=self.config.arm_radius,
                    collision_events=self.collision_events,
                    response_config=response_config,
                    previous_summary=previous_summary,
                    contact_activation_clearance=response_config.contact_release_clearance,
                    summary_segment_slices=solved.summary_segment_slices,
                )
                if optimization_summary.min_clearance < -response_config.optimization_feasibility_tolerance:
                    self._optimization_warm_start = None
                    self._observed_control_slices = None
                    if not response_config.fallback_to_heuristic:
                        raise RuntimeError(
                            "Optimization contact solver produced a penetrating shape: "
                            f"min_clearance={optimization_summary.min_clearance:.3e}"
                        )
                else:
                    self.contact_summary = optimization_summary
                    self.contact_backend_used = "optimization"
                    return

        observed_segments = clone_segments(self.free_segments)
        observed_chain = ContinuumArm2D(observed_segments)
        observed_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        collision_events: list[CollisionEvent] = []
        for _ in range(self.config.max_collision_responses):
            event = find_first_collision(
                segments=observed_chain.segments,
                obstacles=self.obstacles,
                arm_radius=self.config.arm_radius,
                response_config=response_config,
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
                response_config=response_config,
            )
            observed_chain = ContinuumArm2D(observed_segments)
            observed_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        self.observed_segments = observed_chain.segments
        self.observed_chain = observed_chain
        self.collision_events = collision_events
        self._optimization_warm_start = None
        self._observed_control_slices = None
        self.contact_summary = summarize_contacts(
            segments=self.observed_chain.segments,
            obstacles=self.obstacles,
            arm_radius=self.config.arm_radius,
            collision_events=self.collision_events,
            response_config=response_config,
            previous_summary=previous_summary,
        )
        self.contact_backend_used = "heuristic"

    def reset(
        self,
        *,
        obstacles: list[CircleObstacle | tuple[float, float, float]] | None = None,
        base_x: float = 0.0,
        kappas: np.ndarray | None = None,
        tendon_lengths: np.ndarray | None = None,
    ) -> None:
        self.base_x = float(np.clip(base_x, *self.config.base_x_limits))
        self.free_tendon_lengths = self._resolve_initial_tendon_lengths(
            tendon_lengths=tendon_lengths,
            kappas=kappas,
        )
        self.free_kappas = self._kappas_from_tendon_lengths(self.free_tendon_lengths)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.contact_summary = make_empty_contact_summary(len(self.config.segment_lengths))
        self._optimization_warm_start = None
        self._observed_control_slices = None
        self.rebuild()

    def set_obstacles(self, obstacles: list[CircleObstacle | tuple[float, float, float]]) -> None:
        self.obstacles = ensure_obstacles(obstacles)
        self.rebuild()

    def apply_action(self, base_delta: float, tendon_length_deltas: np.ndarray) -> None:
        tendon_length_deltas = np.asarray(tendon_length_deltas, dtype=float)
        if tendon_length_deltas.shape != self.free_tendon_lengths.shape:
            raise ValueError("tendon_length_deltas must have shape (num_segments, 2).")
        self.base_x = float(np.clip(self.base_x + base_delta, *self.config.base_x_limits))
        self.free_tendon_lengths = self._sanitize_tendon_lengths(
            self.free_tendon_lengths + tendon_length_deltas,
        )
        self.free_kappas = self._kappas_from_tendon_lengths(self.free_tendon_lengths)
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

    def backbone_points(self, num_points: int, *, include_root: bool = False) -> np.ndarray:
        return self.observed_chain.sample_keypoints(num_points, include_root=include_root)

    def contact_summary_dict(self) -> dict:
        return self.contact_summary.as_dict()
