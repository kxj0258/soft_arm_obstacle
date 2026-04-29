from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import left_normal, normalize
from common.pcc import ContinuumArm2D, PCCSegment2D

from .physics_2d import CircleObstacle, ContactResponse2DConfig


@dataclass(frozen=True)
class OptimizationContactSolveResult:
    segments: list[PCCSegment2D]
    chain: ContinuumArm2D
    warm_start_kappas: np.ndarray
    summary_segment_slices: tuple[slice, ...]
    reduced_obstacle_count: int


def _build_discretized_profile(
    *,
    segment_lengths: tuple[float, ...],
    segment_kappas: np.ndarray,
    subsegments_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
    lengths: list[float] = []
    kappas: list[float] = []
    slices: list[slice] = []
    cursor = 0
    for segment_length, segment_kappa in zip(segment_lengths, segment_kappas):
        count = max(int(subsegments_per_segment), 2)
        subsegment_length = float(segment_length) / float(count)
        lengths.extend([subsegment_length] * count)
        kappas.extend([float(segment_kappa)] * count)
        slices.append(slice(cursor, cursor + count))
        cursor += count
    return (
        np.asarray(lengths, dtype=float),
        np.asarray(kappas, dtype=float),
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
        end_position = segment.end_position
        end_tangent = normalize(segment.end_direction)
        end_normal = normalize(left_normal(end_tangent))

        midpoint, midpoint_tangent = segment.pose_at_fraction(0.5)
        midpoint_normal = normalize(left_normal(midpoint_tangent))

        points.append(end_position)
        points.append(end_position + arm_radius * end_normal)
        points.append(end_position - arm_radius * end_normal)
        points.append(midpoint + arm_radius * midpoint_normal)
        points.append(midpoint - arm_radius * midpoint_normal)
    return np.asarray(points, dtype=float)


def _select_local_obstacles(
    *,
    obstacles: list[CircleObstacle],
    guessed_chain: ContinuumArm2D,
    arm_radius: float,
    response_config: ContactResponse2DConfig,
) -> list[CircleObstacle]:
    if not obstacles:
        return []
    guessed_points = guessed_chain.sample_points(max(3, response_config.broadphase_samples))
    active_obstacles: list[CircleObstacle] = []
    for obstacle in obstacles:
        threshold = response_config.optimization_locality_margin_factor * max(
            obstacle.radius + arm_radius,
            arm_radius,
        )
        clearance = float(np.min(obstacle.clearance_values(guessed_points, inflation=arm_radius)))
        if clearance <= threshold:
            active_obstacles.append(obstacle)
    return active_obstacles


def solve_optimization_contact_response(
    *,
    segment_lengths: tuple[float, ...],
    segment_kappas: np.ndarray,
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

    discretized_lengths, free_profile, summary_segment_slices = _build_discretized_profile(
        segment_lengths=segment_lengths,
        segment_kappas=np.asarray(segment_kappas, dtype=float),
        subsegments_per_segment=response_config.optimization_subsegments_per_segment,
    )

    if previous_solution is not None and np.asarray(previous_solution).shape == free_profile.shape:
        x0 = np.asarray(previous_solution, dtype=float).copy()
    else:
        x0 = free_profile.copy()
    x0 = np.clip(x0, -curvature_limit, curvature_limit)

    _, guessed_chain = _build_chain(
        lengths=discretized_lengths,
        kappas=x0,
        start_position=np.asarray(start_position, dtype=float),
        start_direction=np.asarray(start_direction, dtype=float),
    )
    reduced_obstacles = _select_local_obstacles(
        obstacles=obstacles,
        guessed_chain=guessed_chain,
        arm_radius=arm_radius,
        response_config=response_config,
    )
    if not reduced_obstacles:
        segments, chain = _build_chain(
            lengths=discretized_lengths,
            kappas=free_profile,
            start_position=np.asarray(start_position, dtype=float),
            start_direction=np.asarray(start_direction, dtype=float),
        )
        return OptimizationContactSolveResult(
            segments=segments,
            chain=chain,
            warm_start_kappas=free_profile.copy(),
            summary_segment_slices=summary_segment_slices,
            reduced_obstacle_count=0,
        )

    def _objective(x: np.ndarray) -> float:
        bend_term = float(np.sum((x * discretized_lengths) ** 2))
        tracking_term = response_config.optimization_tracking_weight * float(np.sum((x - free_profile) ** 2))
        smoothness_term = response_config.optimization_smoothness_weight * float(np.sum(np.diff(x) ** 2))
        return bend_term + tracking_term + smoothness_term

    def _constraint_values(x: np.ndarray) -> np.ndarray:
        _, chain = _build_chain(
            lengths=discretized_lengths,
            kappas=x,
            start_position=np.asarray(start_position, dtype=float),
            start_direction=np.asarray(start_direction, dtype=float),
        )
        points = _constraint_points(chain, arm_radius)
        values = [obstacle.implicit_values(points) for obstacle in reduced_obstacles]
        return np.concatenate(values).astype(float, copy=False)

    constraints: list[dict[str, object]] = []
    for segment_index, segment_slice in enumerate(summary_segment_slices):
        target_kappa = float(segment_kappas[segment_index])
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, segment_slice=segment_slice, target_kappa=target_kappa: float(
                    np.mean(x[segment_slice]) - target_kappa
                ),
            }
        )
    constraints.append({"type": "ineq", "fun": _constraint_values})

    bounds = [(-curvature_limit, curvature_limit)] * int(free_profile.size)
    result = minimize(
        _objective,
        x0,
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
        raise RuntimeError(f"Optimization contact solver failed: {result.message}")

    solved_kappas = np.clip(np.asarray(result.x, dtype=float), -curvature_limit, curvature_limit)
    eq_residual = max(
        abs(float(np.mean(solved_kappas[segment_slice]) - float(segment_kappas[index])))
        for index, segment_slice in enumerate(summary_segment_slices)
    )
    ineq_residual = float(np.min(_constraint_values(solved_kappas)))
    tolerance = float(response_config.optimization_feasibility_tolerance)
    if eq_residual > tolerance or ineq_residual < -tolerance:
        raise RuntimeError(
            "Optimization contact solver returned an infeasible solution: "
            f"eq_residual={eq_residual:.3e}, min_ineq={ineq_residual:.3e}"
        )

    segments, chain = _build_chain(
        lengths=discretized_lengths,
        kappas=solved_kappas,
        start_position=np.asarray(start_position, dtype=float),
        start_direction=np.asarray(start_direction, dtype=float),
    )
    return OptimizationContactSolveResult(
        segments=segments,
        chain=chain,
        warm_start_kappas=solved_kappas,
        summary_segment_slices=summary_segment_slices,
        reduced_obstacle_count=len(reduced_obstacles),
    )
