from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import logging

import numpy as np

from common.pcc import ContinuumArm2D, PCCSegment2D
from common.utils import wrap_angle

from .contact_solver_2d import solve_optimization_contact_response
from .physics_2d import (
    CircleObstacle,
    CollisionEvent,
    ContactResponse2DConfig,
    ContactSummary,
    ensure_obstacles,
    find_collision_events,
    make_empty_contact_summary,
    summarize_contacts,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmModel2DConfig:
    segment_lengths: tuple[float, ...] = (10.0, 10.0)
    base_x_limits: tuple[float, float] = (0.0, 10.0)
    curvature_limit: float = 0.30
    radius_base: float = 0.2
    radius_tip: float | None = None
    tendon_rest_length: float = 10.0
    tendon_routing_radius: float = 0.5
    tendon_length_limits: tuple[float, float] = (5.0, 15.0)
    max_jacobian_iters: int = 5
    max_tendon_delta_per_step: float = 0.02
    min_segment_length: float = 1.0
    render_points_per_segment: int = 32
    contact_response: ContactResponse2DConfig = field(default_factory=ContactResponse2DConfig)
    arm_radius: InitVar[float | None] = None
    arm_radius_tip: InitVar[float | None] = None
    max_collision_responses: InitVar[int | None] = None

    def __post_init__(
        self,
        arm_radius: float | None,
        arm_radius_tip: float | None,
        max_collision_responses: int | None,
    ) -> None:
        radius_base = float(self.radius_base if arm_radius is None else arm_radius)
        if radius_base <= 0.0:
            raise ValueError("radius_base must be positive.")
        radius_tip = self.radius_tip if arm_radius_tip is None else arm_radius_tip
        if radius_tip is None:
            radius_tip = 0.5 * radius_base
        radius_tip = float(np.clip(float(radius_tip), 1e-8, radius_base))
        max_jacobian_iters = self.max_jacobian_iters if max_collision_responses is None else max_collision_responses
        max_jacobian_iters = max(int(max_jacobian_iters), 1)
        max_tendon_delta_per_step = max(float(self.max_tendon_delta_per_step), 1e-8)

        object.__setattr__(self, "radius_base", radius_base)
        object.__setattr__(self, "radius_tip", radius_tip)
        object.__setattr__(self, "max_jacobian_iters", max_jacobian_iters)
        object.__setattr__(self, "max_tendon_delta_per_step", max_tendon_delta_per_step)
        object.__setattr__(self, "arm_radius", radius_base)
        object.__setattr__(self, "arm_radius_tip", radius_tip)
        object.__setattr__(self, "max_collision_responses", max_jacobian_iters)

    def __getattr__(self, name: str) -> object:
        if name == "arm_radius":
            return self.radius_base
        if name == "arm_radius_tip":
            return self.radius_tip
        if name == "max_collision_responses":
            return self.max_jacobian_iters
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


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
        self._rebuild_contact_summary = self.contact_summary.copy()
        self.contact_backend_used = "heuristic"
        self.contact_backend_details: dict[str, object] = {
            "requested_backend": self.config.contact_response.backend,
            "used_backend": "heuristic",
            "status": "initialized",
            "reason": "model_initialized",
        }
        self._optimization_event_logging_enabled = bool(self.config.contact_response.log_optimization_events)
        self._optimization_event_summary = self._make_empty_optimization_event_summary()
        self._optimization_warm_start: np.ndarray | None = None
        self._observed_control_slices: tuple[slice, ...] | None = None
        self.action_blocked = False
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

    def _build_segments_from_kappas(self, kappas: np.ndarray) -> tuple[list[PCCSegment2D], ContinuumArm2D]:
        segments = [
            PCCSegment2D(length=length, kappa=float(kappa))
            for length, kappa in zip(self.config.segment_lengths, np.asarray(kappas, dtype=float))
        ]
        chain = ContinuumArm2D(segments)
        chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)
        return segments, chain

    def _arm_radius_tip(self) -> float:
        tip_radius = self.config.radius_tip
        if tip_radius is None:
            tip_radius = 0.5 * float(self.config.radius_base)
        return float(np.clip(tip_radius, 1e-8, float(self.config.radius_base)))

    def _segment_prefix_lengths(self) -> np.ndarray:
        lengths = np.asarray(self.config.segment_lengths, dtype=float)
        if lengths.size == 0:
            return np.zeros(0, dtype=float)
        return np.concatenate(([0.0], np.cumsum(lengths)[:-1]))

    def _total_arm_length(self) -> float:
        return float(np.sum(np.asarray(self.config.segment_lengths, dtype=float)))

    def _arm_radius_at_fraction(self, fraction: float) -> float:
        clipped_fraction = float(np.clip(fraction, 0.0, 1.0))
        return float(
            float(self.config.radius_base)
            - clipped_fraction * (float(self.config.radius_base) - self._arm_radius_tip())
        )

    def _segment_arm_radius(self, segment_index: int, fraction: float) -> float:
        total_length = max(self._total_arm_length(), 1e-8)
        prefixes = self._segment_prefix_lengths()
        arclength = float(prefixes[segment_index]) + float(np.clip(fraction, 0.0, 1.0)) * float(
            self.config.segment_lengths[segment_index]
        )
        return self._arm_radius_at_fraction(arclength / total_length)

    def _event_global_fraction(self, event: CollisionEvent) -> float:
        total_length = max(self._total_arm_length(), 1e-8)
        prefixes = self._segment_prefix_lengths()
        arclength = float(prefixes[event.segment_index]) + float(np.clip(event.fraction, 0.0, 1.0)) * float(
            self.config.segment_lengths[event.segment_index]
        )
        return float(np.clip(arclength / total_length, 0.0, 1.0))

    def _contact_signed_clearance(
        self,
        *,
        segments: list[PCCSegment2D],
        event: CollisionEvent,
    ) -> float:
        segment = segments[event.segment_index]
        point = segment.point_at(event.fraction)
        clearance = float(np.linalg.norm(point - self.obstacles[event.obstacle_index].center))
        clearance -= float(self.obstacles[event.obstacle_index].radius)
        clearance -= self._segment_arm_radius(event.segment_index, event.fraction)
        return float(clearance)

    def _contact_penetration(
        self,
        *,
        segments: list[PCCSegment2D],
        event: CollisionEvent,
    ) -> float:
        segment = segments[event.segment_index]
        point = segment.point_at(event.fraction)
        obstacle = self.obstacles[event.obstacle_index]
        local_radius = self._arm_radius_at_fraction(self._event_global_fraction(event))
        signed_normal_clearance = float(
            np.dot(point - obstacle.center, event.normal) - (float(obstacle.radius) + local_radius)
        )
        return max(-signed_normal_clearance, 0.0)

    def _contact_penetrations(
        self,
        *,
        segments: list[PCCSegment2D],
        events: list[CollisionEvent],
    ) -> np.ndarray:
        if not events:
            return np.zeros(0, dtype=float)
        return np.asarray([self._contact_penetration(segments=segments, event=event) for event in events], dtype=float)

    def _build_contact_jacobian(
        self,
        *,
        kappas: np.ndarray,
        events: list[CollisionEvent],
        base_penetrations: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        num_controls = int(np.asarray(kappas, dtype=float).size)
        if not events or num_controls == 0:
            return np.zeros((len(events), num_controls), dtype=float)

        curvature_limit = max(float(self.config.curvature_limit), 1e-8)
        jacobian = np.zeros((len(events), num_controls), dtype=float)
        base_kappas = np.asarray(kappas, dtype=float).copy()

        for control_index in range(num_controls):
            perturbed_kappas = base_kappas.copy()
            perturbation = float(epsilon)
            if perturbed_kappas[control_index] + perturbation > curvature_limit:
                perturbation = -perturbation
            perturbed_value = np.clip(
                perturbed_kappas[control_index] + perturbation,
                -curvature_limit,
                curvature_limit,
            )
            if abs(perturbed_value - perturbed_kappas[control_index]) <= 1e-12:
                continue
            perturbed_kappas[control_index] = perturbed_value
            perturbed_segments, _ = self._build_segments_from_kappas(perturbed_kappas)
            perturbed_penetrations = self._contact_penetrations(segments=perturbed_segments, events=events)
            jacobian[:, control_index] = (perturbed_penetrations - base_penetrations) / perturbation

        return jacobian

    def _resolve_simultaneous_contacts(
        self,
        *,
        trial_kappas: np.ndarray,
        response_config: ContactResponse2DConfig,
    ) -> tuple[list[PCCSegment2D], ContinuumArm2D, list[CollisionEvent]]:
        kappas = np.asarray(trial_kappas, dtype=float).copy()
        curvature_limit = float(self.config.curvature_limit)
        epsilon = 1e-4
        damping = 1e-2
        max_iterations = max(int(self.config.max_jacobian_iters), 1)

        best_segments: list[PCCSegment2D] = []
        best_chain = ContinuumArm2D([])
        best_events: list[CollisionEvent] = []
        best_max_penetration = float("inf")

        for _ in range(max_iterations):
            current_segments, current_chain = self._build_segments_from_kappas(kappas)
            events = find_collision_events(
                segments=current_chain.segments,
                obstacles=self.obstacles,
                arm_radius=self.config.radius_base,
                response_config=response_config,
                arm_radius_tip=self._arm_radius_tip(),
            )
            if not events:
                return current_segments, current_chain, []

            penetrations = self._contact_penetrations(segments=current_chain.segments, events=events)
            if penetrations.size == 0:
                return current_segments, current_chain, []

            active_mask = penetrations > float(response_config.collision_epsilon)
            if not np.any(active_mask):
                return current_segments, current_chain, events

            active_events = [events[index] for index in np.flatnonzero(active_mask)]
            active_penetrations = penetrations[active_mask]
            max_penetration = float(np.max(active_penetrations))
            if max_penetration < best_max_penetration:
                best_segments = current_segments
                best_chain = current_chain
                best_events = events
                best_max_penetration = max_penetration

            if max_penetration <= float(response_config.collision_epsilon):
                return current_segments, current_chain, events

            jacobian = self._build_contact_jacobian(
                kappas=kappas,
                events=active_events,
                base_penetrations=active_penetrations,
                epsilon=epsilon,
            )
            if jacobian.size == 0 or not np.any(np.abs(jacobian) > 0.0):
                break

            projection_matrix = jacobian @ jacobian.T + damping * np.eye(len(active_events), dtype=float)
            delta_kappa = jacobian.T @ np.linalg.solve(
                projection_matrix,
                -active_penetrations.reshape(-1, 1),
            )
            delta_kappa = delta_kappa.reshape(-1)
            if not np.all(np.isfinite(delta_kappa)):
                break

            max_step = max(0.25 * curvature_limit, epsilon)
            step_norm = float(np.max(np.abs(delta_kappa))) if delta_kappa.size else 0.0
            if step_norm > max_step:
                delta_kappa = delta_kappa * (max_step / step_norm)

            next_kappas = np.clip(kappas + delta_kappa, -curvature_limit, curvature_limit)
            if np.allclose(next_kappas, kappas, atol=1e-8, rtol=0.0):
                break
            kappas = next_kappas

        final_segments, final_chain = self._build_segments_from_kappas(kappas)
        final_events = find_collision_events(
            segments=final_chain.segments,
            obstacles=self.obstacles,
            arm_radius=self.config.radius_base,
            response_config=response_config,
            arm_radius_tip=self._arm_radius_tip(),
        )
        final_penetrations = self._contact_penetrations(segments=final_chain.segments, events=final_events)
        final_max_penetration = float(np.max(final_penetrations)) if final_penetrations.size else 0.0
        if best_events and best_max_penetration < final_max_penetration:
            return best_segments, best_chain, best_events
        return final_segments, final_chain, final_events

    def _copy_collision_event(self, event: CollisionEvent) -> CollisionEvent:
        return CollisionEvent(
            obstacle_index=int(event.obstacle_index),
            segment_index=int(event.segment_index),
            fraction=float(event.fraction),
            nearest_point=np.asarray(event.nearest_point, dtype=float).copy(),
            contact_point=np.asarray(event.contact_point, dtype=float).copy(),
            tangent=np.asarray(event.tangent, dtype=float).copy(),
            normal=np.asarray(event.normal, dtype=float).copy(),
            penetration=float(event.penetration),
        )

    def _merge_collision_events(
        self,
        existing_events: list[CollisionEvent],
        new_events: list[CollisionEvent],
    ) -> list[CollisionEvent]:
        merged: dict[tuple[int, int, int], CollisionEvent] = {}
        for event in existing_events:
            key = (int(event.obstacle_index), int(event.segment_index), int(round(float(event.fraction) * 1000.0)))
            merged[key] = self._copy_collision_event(event)
        for event in new_events:
            key = (int(event.obstacle_index), int(event.segment_index), int(round(float(event.fraction) * 1000.0)))
            current = merged.get(key)
            if current is None or float(event.penetration) > float(current.penetration):
                merged[key] = self._copy_collision_event(event)
        return sorted(
            merged.values(),
            key=lambda event: (-float(event.penetration), int(event.segment_index), float(event.fraction), int(event.obstacle_index)),
        )

    def _aggregate_contact_summaries(
        self,
        summaries: list[ContactSummary],
        collision_events: list[CollisionEvent],
    ) -> ContactSummary:
        if not summaries:
            return make_empty_contact_summary(len(self.config.segment_lengths))

        state_priority = {"free": 0, "constrained": 1, "latched": 2, "collision": 3}
        reference = min(
            summaries,
            key=lambda summary: (
                float(summary.min_clearance),
                -float(summary.primary_penetration),
            ),
        )
        segment_clearances = np.full(reference.segment_clearances.shape, float("inf"), dtype=float)
        segment_contact_mask = np.zeros(reference.segment_contact_mask.shape, dtype=bool)
        for summary in summaries:
            segment_clearances = np.minimum(segment_clearances, np.asarray(summary.segment_clearances, dtype=float))
            segment_contact_mask |= np.asarray(summary.segment_contact_mask, dtype=bool)

        contact_state = max(
            summaries,
            key=lambda summary: state_priority.get(summary.contact_state, 0),
        ).contact_state
        if collision_events:
            contact_state = "collision"

        return ContactSummary(
            active=bool(any(summary.active for summary in summaries) or collision_events),
            contact_state=contact_state,
            collision_count=len(collision_events) if collision_events else max(summary.collision_count for summary in summaries),
            min_clearance=float(reference.min_clearance),
            nearest_obstacle_index=int(reference.nearest_obstacle_index),
            nearest_segment_index=int(reference.nearest_segment_index),
            nearest_fraction=float(reference.nearest_fraction),
            nearest_arm_point=np.asarray(reference.nearest_arm_point, dtype=float).copy(),
            nearest_obstacle_point=np.asarray(reference.nearest_obstacle_point, dtype=float).copy(),
            nearest_obstacle_center=np.asarray(reference.nearest_obstacle_center, dtype=float).copy(),
            nearest_obstacle_radius=float(reference.nearest_obstacle_radius),
            primary_penetration=max(
                max((float(summary.primary_penetration) for summary in summaries), default=0.0),
                max((float(event.penetration) for event in collision_events), default=0.0),
            ),
            segment_clearances=segment_clearances,
            segment_contact_mask=segment_contact_mask,
        )

    def _severe_penetration_threshold(self) -> float:
        return max(
            10.0 * float(self.config.contact_response.collision_epsilon),
            0.25 * min(float(self.config.radius_base), self._arm_radius_tip()),
        )

    def _has_severe_penetration(self, summary: ContactSummary) -> bool:
        residual_penetration = max(float(summary.primary_penetration), max(-float(summary.min_clearance), 0.0))
        return residual_penetration >= self._severe_penetration_threshold()

    def _make_empty_optimization_event_summary(self) -> dict[str, object]:
        return {
            "event_count": 0,
            "fallback_count": 0,
            "solver_failure_count": 0,
            "penetrating_shape_count": 0,
            "worst_min_clearance": float("inf"),
            "last_reason": "",
        }

    def reset_optimization_event_summary(self) -> None:
        self._optimization_event_summary = self._make_empty_optimization_event_summary()

    def consume_optimization_event_summary(self, *, reset: bool = True) -> dict[str, object]:
        summary = dict(self._optimization_event_summary)
        if reset:
            self.reset_optimization_event_summary()
        return summary

    def set_optimization_event_logging(self, enabled: bool) -> None:
        self._optimization_event_logging_enabled = bool(enabled)

    def _record_optimization_event(
        self,
        *,
        category: str,
        reason: str,
        min_clearance: float | None = None,
    ) -> None:
        summary = self._optimization_event_summary
        summary["event_count"] = int(summary["event_count"]) + 1
        summary["fallback_count"] = int(summary["fallback_count"]) + 1
        if category == "solver_failure":
            summary["solver_failure_count"] = int(summary["solver_failure_count"]) + 1
        elif category == "penetrating_shape":
            summary["penetrating_shape_count"] = int(summary["penetrating_shape_count"]) + 1
        summary["last_reason"] = str(reason)
        if min_clearance is not None and np.isfinite(float(min_clearance)):
            current_worst = float(summary["worst_min_clearance"])
            summary["worst_min_clearance"] = min(current_worst, float(min_clearance))
        if self._optimization_event_logging_enabled:
            LOGGER.warning("%s", reason)

    def rebuild(self) -> None:
        previous_summary = self._rebuild_contact_summary
        optimization_error: Exception | None = None
        optimization_diagnostics: dict[str, object] = {}

        self.free_segments = self._build_free_segments()
        self.free_chain = ContinuumArm2D(self.free_segments)
        self.free_chain.update_chain(start_position=self.root_position, start_direction=self.root_direction)

        response_config = self.config.contact_response
        if response_config.backend == "optimization":
            try:
                solved = solve_optimization_contact_response(
                    segment_lengths=self.config.segment_lengths,
                    segment_kappas=self.free_kappas,
                    segment_tendon_lengths=self.free_tendon_lengths,
                    tendon_rest_length=self.config.tendon_rest_length,
                    tendon_routing_radius=self.config.tendon_routing_radius,
                    obstacles=self.obstacles,
                    arm_radius=self.config.radius_base,
                    curvature_limit=self.config.curvature_limit,
                    start_position=self.root_position,
                    start_direction=self.root_direction,
                    response_config=response_config,
                    previous_solution=self._optimization_warm_start,
                )
            except (ImportError, RuntimeError) as exc:
                optimization_error = exc
                self._record_optimization_event(
                    category="solver_failure",
                    reason=f"Optimization contact backend falling back to heuristic: {exc}",
                )
                if not response_config.fallback_to_heuristic:
                    raise
            else:
                self.observed_segments = solved.segments
                self.observed_chain = solved.chain
                self.collision_events = []
                self._optimization_warm_start = solved.warm_start_kappas.copy()
                self._observed_control_slices = solved.summary_segment_slices
                optimization_diagnostics = {
                    "optimization_message": solved.diagnostics.message,
                    "constraint_mode": solved.diagnostics.constraint_mode,
                    "attempt_count": int(solved.diagnostics.attempt_count),
                    "selected_start_label": solved.diagnostics.selected_start_label,
                    "reduced_obstacle_count": int(solved.diagnostics.reduced_obstacle_count),
                    "iterations": int(solved.diagnostics.iterations),
                    "objective_value": float(solved.diagnostics.objective_value),
                    "max_tendon_residual": float(solved.diagnostics.max_tendon_residual),
                    "max_tendon_differential_residual": float(solved.diagnostics.max_tendon_differential_residual),
                    "min_clearance_residual": float(solved.diagnostics.min_clearance_residual),
                    "tendon_residuals": solved.diagnostics.tendon_residuals.copy(),
                    "tendon_differential_residuals": solved.diagnostics.tendon_differential_residuals.copy(),
                }
                optimization_summary = summarize_contacts(
                    segments=self.observed_chain.segments,
                    obstacles=self.obstacles,
                    arm_radius=self.config.radius_base,
                    collision_events=self.collision_events,
                    response_config=response_config,
                    previous_summary=previous_summary,
                    arm_radius_tip=self._arm_radius_tip(),
                    contact_activation_clearance=response_config.contact_release_clearance,
                    summary_segment_slices=solved.summary_segment_slices,
                )
                if optimization_summary.min_clearance < -response_config.optimization_feasibility_tolerance:
                    self._optimization_warm_start = None
                    self._observed_control_slices = None
                    optimization_error = RuntimeError(
                        "Optimization contact solver produced a penetrating shape: "
                        f"min_clearance={optimization_summary.min_clearance:.3e}"
                    )
                    self._record_optimization_event(
                        category="penetrating_shape",
                        reason=str(optimization_error),
                        min_clearance=float(optimization_summary.min_clearance),
                    )
                    if not response_config.fallback_to_heuristic:
                        raise optimization_error
                else:
                    self.contact_summary = optimization_summary
                    self._rebuild_contact_summary = optimization_summary.copy()
                    self.contact_backend_used = "optimization"
                    self.contact_backend_details = {
                        "requested_backend": "optimization",
                        "used_backend": "optimization",
                        "status": "active",
                        "reason": solved.diagnostics.message,
                        "contact_state": optimization_summary.contact_state,
                        **optimization_diagnostics,
                    }
                    return

        observed_segments, observed_chain, collision_events = self._resolve_simultaneous_contacts(
            trial_kappas=self.free_kappas,
            response_config=response_config,
        )

        self.observed_segments = observed_chain.segments
        self.observed_chain = observed_chain
        self.collision_events = collision_events
        self._optimization_warm_start = None
        self._observed_control_slices = None
        self.contact_summary = summarize_contacts(
            segments=self.observed_chain.segments,
            obstacles=self.obstacles,
            arm_radius=self.config.radius_base,
            collision_events=self.collision_events,
            response_config=response_config,
            previous_summary=previous_summary,
            arm_radius_tip=self._arm_radius_tip(),
        )
        self._rebuild_contact_summary = self.contact_summary.copy()
        self.contact_backend_used = "heuristic"
        if response_config.backend == "optimization":
            self.contact_backend_details = {
                "requested_backend": "optimization",
                "used_backend": "heuristic",
                "status": "fallback",
                "reason": str(optimization_error) if optimization_error is not None else "optimization_unavailable",
                "contact_state": self.contact_summary.contact_state,
                **optimization_diagnostics,
            }
        else:
            self.contact_backend_details = {
                "requested_backend": "heuristic",
                "used_backend": "heuristic",
                "status": "active",
                "reason": "heuristic_contact_response",
                "contact_state": self.contact_summary.contact_state,
            }

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
        self._rebuild_contact_summary = self.contact_summary.copy()
        self.reset_optimization_event_summary()
        self._optimization_warm_start = None
        self._observed_control_slices = None
        self.action_blocked = False
        self.rebuild()

    def snapshot(self) -> dict[str, object]:
        return {
            "base_x": float(self.base_x),
            "free_tendon_lengths": self.free_tendon_lengths.copy(),
            "obstacles": ensure_obstacles(self.obstacles),
            "contact_summary": self.contact_summary.copy(),
            "action_blocked": bool(self.action_blocked),
            "optimization_warm_start": None
            if self._optimization_warm_start is None
            else self._optimization_warm_start.copy(),
            "observed_control_slices": None
            if self._observed_control_slices is None
            else tuple(slice(item.start, item.stop, item.step) for item in self._observed_control_slices),
        }

    def restore(self, state: dict[str, object]) -> None:
        self.base_x = float(np.clip(float(state["base_x"]), *self.config.base_x_limits))
        self.free_tendon_lengths = self._sanitize_tendon_lengths(
            np.asarray(state["free_tendon_lengths"], dtype=float),
        )
        self.free_kappas = self._kappas_from_tendon_lengths(self.free_tendon_lengths)
        self.obstacles = ensure_obstacles(state.get("obstacles", []))
        contact_summary = state.get("contact_summary")
        if isinstance(contact_summary, ContactSummary):
            self.contact_summary = contact_summary.copy()
        else:
            self.contact_summary = make_empty_contact_summary(len(self.config.segment_lengths))
        self._rebuild_contact_summary = self.contact_summary.copy()
        self.action_blocked = bool(state.get("action_blocked", False))
        warm_start = state.get("optimization_warm_start")
        self._optimization_warm_start = None if warm_start is None else np.asarray(warm_start, dtype=float).copy()
        observed_control_slices = state.get("observed_control_slices")
        self._observed_control_slices = (
            None
            if observed_control_slices is None
            else tuple(slice(item.start, item.stop, item.step) for item in observed_control_slices)
        )
        self.rebuild()

    def set_obstacles(self, obstacles: list[CircleObstacle | tuple[float, float, float]]) -> None:
        self.obstacles = ensure_obstacles(obstacles)
        self.rebuild()

    def apply_action(self, base_delta: float, tendon_length_deltas: np.ndarray) -> None:
        tendon_length_deltas = np.asarray(tendon_length_deltas, dtype=float)
        if tendon_length_deltas.shape != self.free_tendon_lengths.shape:
            raise ValueError("tendon_length_deltas must have shape (num_segments, 2).")
        base_delta = float(base_delta)
        max_displacement = float(
            max(
                abs(base_delta),
                float(np.max(np.abs(tendon_length_deltas))) if tendon_length_deltas.size else 0.0,
            )
        )
        sub_steps = max(1, int(np.ceil(max_displacement / float(self.config.max_tendon_delta_per_step))))
        base_step = base_delta / float(sub_steps)
        tendon_step = tendon_length_deltas / float(sub_steps)

        current_base_x = float(self.base_x)
        current_tendon_lengths = self.free_tendon_lengths.copy()
        safe_base_x = current_base_x
        safe_tendon_lengths = current_tendon_lengths.copy()
        safe_kappas = self.free_kappas.copy()
        self.action_blocked = False

        for _ in range(sub_steps):
            current_base_x = float(np.clip(current_base_x + base_step, *self.config.base_x_limits))
            current_tendon_lengths = self._sanitize_tendon_lengths(current_tendon_lengths + tendon_step)
            self.base_x = current_base_x
            self.free_tendon_lengths = current_tendon_lengths.copy()
            self.free_kappas = self._kappas_from_tendon_lengths(self.free_tendon_lengths)
            self.rebuild()
            if self._has_severe_penetration(self.contact_summary):
                self.base_x = safe_base_x
                self.free_tendon_lengths = safe_tendon_lengths.copy()
                self.free_kappas = safe_kappas.copy()
                self.rebuild()
                current_base_x = safe_base_x
                current_tendon_lengths = safe_tendon_lengths.copy()
                self.action_blocked = True
                break
            safe_base_x = current_base_x
            safe_tendon_lengths = self.free_tendon_lengths.copy()
            safe_kappas = self.free_kappas.copy()

        self.base_x = safe_base_x
        self.free_tendon_lengths = safe_tendon_lengths.copy()
        self.free_kappas = safe_kappas.copy()
        self.contact_backend_details = {
            **self.contact_backend_details,
            "contact_state": self.contact_summary.contact_state,
            "sub_steps": sub_steps,
            "action_blocked": self.action_blocked,
            "aggregated_collision_count": len(self.collision_events),
        }

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
