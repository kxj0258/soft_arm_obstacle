from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import left_normal, normalize
from common.pcc import ContinuumArm2D, PCCSegment2D

from .physics_2d import CircleObstacle, ContactResponse2DConfig


@dataclass(frozen=True)
class OptimizationContactDiagnostics:
    success: bool
    message: str
    constraint_mode: str
    attempt_count: int
    selected_start_label: str
    reduced_obstacle_count: int
    iterations: int
    objective_value: float
    max_tendon_residual: float
    max_tendon_differential_residual: float
    min_clearance_residual: float
    tendon_residuals: np.ndarray
    tendon_differential_residuals: np.ndarray


@dataclass(frozen=True)
class OptimizationContactSolveResult:
    segments: list[PCCSegment2D]
    chain: ContinuumArm2D
    warm_start_kappas: np.ndarray
    summary_segment_slices: tuple[slice, ...]
    reduced_obstacle_count: int
    diagnostics: OptimizationContactDiagnostics


def _build_discretized_profile(
    *,
    segment_lengths: tuple[float, ...],
    segment_kappas: np.ndarray,
    segment_neutral_tendon_lengths: np.ndarray,
    subsegments_per_segment: int,
    tendon_rest_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, ...]]:
    lengths: list[float] = []
    kappas: list[float] = []
    rest_lengths: list[float] = []
    slices: list[slice] = []
    cursor = 0
    for segment_index, (segment_length, segment_kappa) in enumerate(zip(segment_lengths, segment_kappas)):
        count = max(int(subsegments_per_segment), 2)
        safe_segment_length = max(float(segment_length), 1e-8)
        subsegment_length = float(segment_length) / float(count)
        neutral_tendon_length = float(segment_neutral_tendon_lengths[segment_index])
        if not np.isfinite(neutral_tendon_length) or neutral_tendon_length <= 0.0:
            neutral_tendon_length = float(tendon_rest_length)
        subsegment_rest_length = neutral_tendon_length * (subsegment_length / safe_segment_length)
        lengths.extend([subsegment_length] * count)
        kappas.extend([float(segment_kappa)] * count)
        rest_lengths.extend([subsegment_rest_length] * count)
        slices.append(slice(cursor, cursor + count))
        cursor += count
    return (
        np.asarray(lengths, dtype=float),
        np.asarray(kappas, dtype=float),
        np.asarray(rest_lengths, dtype=float),
        tuple(slices),
    )


def _build_chain(
    *,
    lengths: np.ndarray,
    kappas: np.ndarray,
    start_position: np.ndarray,
    start_direction: np.ndarray,
) -> tuple[list[PCCSegment2D], ContinuumArm2D]:
    segments = [
        PCCSegment2D(length=float(length), kappa=float(kappa))
        for length, kappa in zip(lengths, kappas)
    ]
    chain = ContinuumArm2D(segments)
    chain.update_chain(start_position=start_position, start_direction=start_direction)
    return segments, chain


def _constraint_points(chain: ContinuumArm2D, arm_radius: float) -> np.ndarray:
    points: list[np.ndarray] = []
    for segment in chain.segments:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            position, tangent = segment.pose_at_fraction(float(fraction))
            normal = normalize(left_normal(tangent))
            points.append(position)
            points.append(position + arm_radius * normal)
            points.append(position - arm_radius * normal)
    return np.asarray(points, dtype=float)


def _select_local_obstacles(
    *,
    obstacles: list[CircleObstacle],
    guessed_chains: list[ContinuumArm2D],
    arm_radius: float,
    response_config: ContactResponse2DConfig,
) -> list[CircleObstacle]:
    if not obstacles:
        return []
    active_obstacles: list[CircleObstacle] = []
    for obstacle in obstacles:
        threshold = response_config.optimization_locality_margin_factor * max(
            obstacle.radius + arm_radius,
            arm_radius,
        )
        nearby = False
        for guessed_chain in guessed_chains:
            guessed_points = guessed_chain.sample_points(max(3, response_config.broadphase_samples))
            clearance = float(np.min(obstacle.clearance_values(guessed_points, inflation=arm_radius)))
            if clearance <= threshold:
                nearby = True
                break
        if nearby:
            active_obstacles.append(obstacle)
    return active_obstacles


def _subsegment_tendon_lengths(
    *,
    lengths: np.ndarray,
    kappas: np.ndarray,
    tendon_rest_lengths: np.ndarray,
    tendon_routing_radius: float,
) -> np.ndarray:
    routing_radius = max(float(tendon_routing_radius), 1e-8)
    left = tendon_rest_lengths - routing_radius * lengths * kappas
    right = tendon_rest_lengths + routing_radius * lengths * kappas
    return np.column_stack((left, right)).astype(float, copy=False)


def solve_optimization_contact_response(
    *,
    segment_lengths: tuple[float, ...],
    segment_kappas: np.ndarray,
    segment_tendon_lengths: np.ndarray,
    tendon_rest_length: float,
    tendon_routing_radius: float,
    obstacles: list[CircleObstacle],
    arm_radius: float,
    curvature_limit: float,
    start_position: np.ndarray,
    start_direction: np.ndarray,
    response_config: ContactResponse2DConfig,
    previous_solution: np.ndarray | None = None,
) -> OptimizationContactSolveResult:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "Optimization contact backend requires scipy. "
            "Install it with the project environment or switch contact_response.backend='heuristic'."
        ) from exc

    target_tendon_lengths = np.asarray(segment_tendon_lengths, dtype=float)
    if target_tendon_lengths.ndim == 1 and target_tendon_lengths.size == 2 * len(segment_lengths):
        target_tendon_lengths = target_tendon_lengths.reshape(len(segment_lengths), 2)
    if target_tendon_lengths.shape != (len(segment_lengths), 2):
        raise ValueError("segment_tendon_lengths must have shape (num_segments, 2).")
    segment_neutral_tendon_lengths = np.mean(target_tendon_lengths, axis=1)
    target_tendon_differentials = target_tendon_lengths[:, 1] - target_tendon_lengths[:, 0]

    discretized_lengths, free_profile, discretized_rest_lengths, summary_segment_slices = _build_discretized_profile(
        segment_lengths=segment_lengths,
        segment_kappas=np.asarray(segment_kappas, dtype=float),
        segment_neutral_tendon_lengths=segment_neutral_tendon_lengths,
        subsegments_per_segment=response_config.optimization_subsegments_per_segment,
        tendon_rest_length=tendon_rest_length,
    )

    trial_starts: list[tuple[str, np.ndarray]] = []

    def _append_unique_start(label: str, candidate: np.ndarray) -> None:
        clipped = np.clip(np.asarray(candidate, dtype=float), -curvature_limit, curvature_limit)
        for _, existing in trial_starts:
            if np.allclose(existing, clipped, atol=1e-6, rtol=1e-6):
                return
        trial_starts.append((label, clipped))

    if previous_solution is not None and np.asarray(previous_solution).shape == free_profile.shape:
        _append_unique_start("previous_solution", np.asarray(previous_solution, dtype=float))
    _append_unique_start("free_profile", free_profile.copy())
    _append_unique_start("blended_profile", 0.5 * (free_profile + trial_starts[0][1]))
    _append_unique_start("zero_profile", np.zeros_like(free_profile))
    _append_unique_start("scaled_profile", 0.75 * free_profile)
    max_attempts = max(int(response_config.optimization_multistart_attempts), 1)
    trial_starts = trial_starts[:max_attempts]

    guessed_chains: list[ContinuumArm2D] = []
    for _, start_guess in trial_starts:
        _, guessed_chain = _build_chain(
            lengths=discretized_lengths,
            kappas=start_guess,
            start_position=np.asarray(start_position, dtype=float),
            start_direction=np.asarray(start_direction, dtype=float),
        )
        guessed_chains.append(guessed_chain)
    reduced_obstacles = _select_local_obstacles(
        obstacles=obstacles,
        guessed_chains=guessed_chains,
        arm_radius=arm_radius,
        response_config=response_config,
    )

    def _tendon_constraint_residuals(x: np.ndarray) -> np.ndarray:
        subsegment_tendon_lengths = _subsegment_tendon_lengths(
            lengths=discretized_lengths,
            kappas=x,
            tendon_rest_lengths=discretized_rest_lengths,
            tendon_routing_radius=tendon_routing_radius,
        )
        residuals = np.zeros_like(target_tendon_lengths, dtype=float)
        for segment_index, segment_slice in enumerate(summary_segment_slices):
            totals = np.sum(subsegment_tendon_lengths[segment_slice], axis=0)
            residuals[segment_index] = totals - target_tendon_lengths[segment_index]
        return residuals

    def _tendon_differential_residuals(x: np.ndarray) -> np.ndarray:
        tendon_residuals = _tendon_constraint_residuals(x)
        return tendon_residuals[:, 1] - tendon_residuals[:, 0]

    def _base_objective(x: np.ndarray) -> float:
        bend_term = float(np.sum((x * discretized_lengths) ** 2))
        tracking_term = response_config.optimization_tracking_weight * float(np.sum((x - free_profile) ** 2))
        smoothness_term = response_config.optimization_smoothness_weight * float(np.sum(np.diff(x) ** 2))
        return bend_term + tracking_term + smoothness_term

    def _constraint_chain(x: np.ndarray) -> tuple[ContinuumArm2D, np.ndarray]:
        _, chain = _build_chain(
            lengths=discretized_lengths,
            kappas=x,
            start_position=np.asarray(start_position, dtype=float),
            start_direction=np.asarray(start_direction, dtype=float),
        )
        points = _constraint_points(chain, arm_radius)
        return chain, points

    if not reduced_obstacles:
        segments, chain = _build_chain(
            lengths=discretized_lengths,
            kappas=free_profile,
            start_position=np.asarray(start_position, dtype=float),
            start_direction=np.asarray(start_direction, dtype=float),
        )
        tendon_residuals = _tendon_constraint_residuals(free_profile)
        tendon_differential_residuals = _tendon_differential_residuals(free_profile)
        diagnostics = OptimizationContactDiagnostics(
            success=True,
            message="no_local_obstacles_near_free_shape",
            constraint_mode="segment_tendon_length_differential",
            attempt_count=1,
            selected_start_label="free_profile",
            reduced_obstacle_count=0,
            iterations=0,
            objective_value=_base_objective(free_profile),
            max_tendon_residual=float(np.max(np.abs(tendon_residuals))),
            max_tendon_differential_residual=float(np.max(np.abs(tendon_differential_residuals))),
            min_clearance_residual=float("inf"),
            tendon_residuals=tendon_residuals.copy(),
            tendon_differential_residuals=tendon_differential_residuals.copy(),
        )
        return OptimizationContactSolveResult(
            segments=segments,
            chain=chain,
            warm_start_kappas=free_profile.copy(),
            summary_segment_slices=summary_segment_slices,
            reduced_obstacle_count=0,
            diagnostics=diagnostics,
        )

    def _constraint_values(x: np.ndarray) -> np.ndarray:
        _, points = _constraint_chain(x)
        values = [obstacle.implicit_values(points) for obstacle in reduced_obstacles]
        return np.concatenate(values).astype(float, copy=False)

    def _constraint_clearances(x: np.ndarray) -> np.ndarray:
        _, points = _constraint_chain(x)
        clearances = [
            obstacle.clearance_values(points, inflation=arm_radius)
            for obstacle in reduced_obstacles
        ]
        return np.concatenate(clearances).astype(float, copy=False)

    def _objective(x: np.ndarray) -> float:
        clearance_violations = np.maximum(
            float(response_config.optimization_clearance_margin) - _constraint_clearances(x),
            0.0,
        )
        violation_penalty = response_config.optimization_violation_penalty_weight * float(np.sum(clearance_violations ** 2))
        return _base_objective(x) + violation_penalty

    constraints: list[dict[str, object]] = []
    for segment_index in range(target_tendon_lengths.shape[0]):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, segment_index=segment_index: float(
                    _tendon_differential_residuals(x)[segment_index]
                ),
            }
        )
    constraints.append({"type": "ineq", "fun": _constraint_values})

    bounds = [(-curvature_limit, curvature_limit)] * int(free_profile.size)
    best_solution: tuple[np.ndarray, object, str] | None = None
    failure_messages: list[str] = []
    tolerance = float(response_config.optimization_feasibility_tolerance)
    for attempt_index, (start_label, start_guess) in enumerate(trial_starts, start=1):
        result = minimize(
            _objective,
            start_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": int(response_config.optimization_max_iterations),
                "ftol": 1e-9,
                "disp": False,
            },
        )
        if not result.success:
            failure_messages.append(
                f"attempt={attempt_index} start={start_label} failure={result.message} "
                f"(status={getattr(result, 'status', 'n/a')}, iterations={getattr(result, 'nit', 0)})"
            )
            continue

        solved_kappas = np.clip(np.asarray(result.x, dtype=float), -curvature_limit, curvature_limit)
        tendon_residuals = _tendon_constraint_residuals(solved_kappas)
        tendon_differential_residuals = _tendon_differential_residuals(solved_kappas)
        eq_residual = float(np.max(np.abs(tendon_differential_residuals)))
        ineq_residual = float(np.min(_constraint_values(solved_kappas)))
        min_constraint_clearance = float(np.min(_constraint_clearances(solved_kappas)))
        if (
            eq_residual > tolerance
            or ineq_residual < -tolerance
            or min_constraint_clearance < -tolerance
        ):
            failure_messages.append(
                "attempt="
                f"{attempt_index} start={start_label} infeasible="
                f"max_tendon_differential_residual={eq_residual:.3e}, "
                f"max_tendon_residual={float(np.max(np.abs(tendon_residuals))):.3e}, "
                f"min_ineq={ineq_residual:.3e}, "
                f"min_constraint_clearance={min_constraint_clearance:.3e}"
            )
            continue

        if best_solution is None or _objective(solved_kappas) < _objective(best_solution[0]):
            best_solution = (solved_kappas, result, start_label)

    if best_solution is None:
        raise RuntimeError(
            "Optimization contact solver failed across all starts: "
            + "; ".join(failure_messages)
            + f" (reduced_obstacles={len(reduced_obstacles)})"
        )

    solved_kappas, result, selected_start_label = best_solution
    tendon_residuals = _tendon_constraint_residuals(solved_kappas)
    tendon_differential_residuals = _tendon_differential_residuals(solved_kappas)
    eq_residual = float(np.max(np.abs(tendon_differential_residuals)))
    ineq_residual = float(np.min(_constraint_values(solved_kappas)))

    segments, chain = _build_chain(
        lengths=discretized_lengths,
        kappas=solved_kappas,
        start_position=np.asarray(start_position, dtype=float),
        start_direction=np.asarray(start_direction, dtype=float),
    )
    diagnostics = OptimizationContactDiagnostics(
        success=True,
        message="optimized_contact_shape",
        constraint_mode="segment_tendon_length_differential",
        attempt_count=len(trial_starts),
        selected_start_label=selected_start_label,
        reduced_obstacle_count=len(reduced_obstacles),
        iterations=int(getattr(result, "nit", 0)),
        objective_value=float(_objective(solved_kappas)),
        max_tendon_residual=float(np.max(np.abs(tendon_residuals))),
        max_tendon_differential_residual=eq_residual,
        min_clearance_residual=ineq_residual,
        tendon_residuals=tendon_residuals.copy(),
        tendon_differential_residuals=tendon_differential_residuals.copy(),
    )
    return OptimizationContactSolveResult(
        segments=segments,
        chain=chain,
        warm_start_kappas=solved_kappas,
        summary_segment_slices=summary_segment_slices,
        reduced_obstacle_count=len(reduced_obstacles),
        diagnostics=diagnostics,
    )
