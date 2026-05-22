from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.frames import rodrigues
from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

from .base import TDCRSolverState
from .pcc_3d import PCC3DSolver, summarize_shape_contacts


@dataclass
class PCCAContact3DSolver:
    robot_params: TDCRRobotParams
    tracking_weight: float = 0.35
    smoothness_weight: float = 0.05
    clearance_margin: float = 1e-4
    max_iterations: int = 80
    fallback_to_free: bool = True
    contact_stiffness: float = 120.0

    def __post_init__(self) -> None:
        self._free_solver = PCC3DSolver(self.robot_params, contact_stiffness=self.contact_stiffness)

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        obstacles = obstacles or []
        free_shape = self._free_solver.solve(tau=tau, obstacles=obstacles, state=state)
        if not obstacles or free_shape.contact_summary.min_clearance >= self.clearance_margin:
            free_shape.diagnostics = {
                **dict(free_shape.diagnostics),
                "solver": "pcca_contact_3d",
                "status": "free_shape_feasible",
                "iterations": 0,
            }
            return free_shape

        try:
            solved_shape, warm_start, diagnostics = self._solve_with_scipy(
                free_shape=free_shape,
                obstacles=obstacles,
                previous_solution=None if state is None else state.warm_start,
            )
        except Exception as exc:
            if not self.fallback_to_free:
                raise
            free_shape.diagnostics = {
                **dict(free_shape.diagnostics),
                "solver": "pcca_contact_3d",
                "status": "fallback",
                "reason": str(exc),
            }
            return free_shape

        if state is not None:
            state.warm_start = warm_start.copy()
        solved_shape.contact_summary = summarize_shape_contacts(
            shape=solved_shape,
            obstacles=obstacles,
            arm_radius=self.robot_params.arm_radius,
            num_segments=self.robot_params.num_segments,
            contact_stiffness=self.contact_stiffness,
        )
        solved_shape.diagnostics = {
            **diagnostics,
            "solver": "pcca_contact_3d",
            "free_min_clearance": float(free_shape.contact_summary.min_clearance),
            "contact_state": solved_shape.contact_summary.contact_state,
        }
        return solved_shape

    def _solve_with_scipy(
        self,
        *,
        free_shape: TDCRShape,
        obstacles: list[Obstacle3D],
        previous_solution: object | None,
    ) -> tuple[TDCRShape, np.ndarray, dict]:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise RuntimeError("PCCAContact3DSolver requires scipy.") from exc

        free_u = self._segment_u_from_shape(free_shape)
        x0 = free_u.reshape(-1)
        if isinstance(previous_solution, np.ndarray) and previous_solution.shape == x0.shape:
            x0 = previous_solution.astype(float, copy=True)
        num_segments = self.robot_params.num_segments
        limit = float(self.robot_params.curvature_limit)
        bounds = [(-limit, limit)] * int(x0.size)

        def build_shape(x: np.ndarray) -> TDCRShape:
            segment_u = np.asarray(x, dtype=float).reshape(num_segments, 3)
            return self._integrate_with_segment_u(segment_u, free_shape.tau)

        def objective(x: np.ndarray) -> float:
            segment_u = np.asarray(x, dtype=float).reshape(num_segments, 3)
            lengths = np.asarray(self.robot_params.segment_lengths, dtype=float)[:, None]
            bend = float(np.sum((segment_u * lengths) ** 2))
            track = self.tracking_weight * float(np.sum((segment_u - free_u) ** 2))
            smooth = self.smoothness_weight * float(np.sum(np.diff(segment_u, axis=0) ** 2))
            return bend + track + smooth

        def clearance_constraint(x: np.ndarray) -> np.ndarray:
            shape = build_shape(x)
            values = []
            for obstacle in obstacles:
                clearances = obstacle.clearance(shape.p, inflation=self.robot_params.arm_radius)
                values.append(clearances - float(self.clearance_margin))
            return np.concatenate(values).astype(float, copy=False)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": clearance_constraint}],
            options={"maxiter": int(self.max_iterations), "ftol": 1e-9, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"3D PCCA contact optimization failed: {result.message}")
        solved_x = np.asarray(result.x, dtype=float)
        solved = build_shape(solved_x)
        min_constraint = float(np.min(clearance_constraint(solved_x)))
        if min_constraint < -1e-5:
            raise RuntimeError(f"3D PCCA produced infeasible clearance residual {min_constraint:.3e}")
        diagnostics = {
            "status": "optimized_contact_shape",
            "iterations": int(getattr(result, "nit", 0)),
            "objective_value": float(objective(solved_x)),
            "min_clearance_residual": min_constraint,
        }
        return solved, solved_x, diagnostics

    def _segment_u_from_shape(self, shape: TDCRShape) -> np.ndarray:
        values = np.zeros((self.robot_params.num_segments, 3), dtype=float)
        for segment_index in range(self.robot_params.num_segments):
            mask = shape.segment_id == segment_index
            if np.any(mask):
                values[segment_index] = np.mean(shape.u[mask], axis=0)
        return values

    def _integrate_with_segment_u(self, segment_u: np.ndarray, tau: np.ndarray) -> TDCRShape:
        positions = [np.zeros(3, dtype=float)]
        rotations = [np.eye(3, dtype=float)]
        u_values = [segment_u[0].copy()]
        v_values = [np.array([0.0, 0.0, 1.0], dtype=float)]
        s_values = [0.0]
        segment_ids = [0]
        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        current_s = 0.0
        for segment_index, length in enumerate(self.robot_params.segment_lengths):
            steps = max(int(self.robot_params.samples_per_segment), 2)
            ds = float(length) / float(steps)
            u = segment_u[segment_index]
            for _ in range(steps):
                half_rotation = rotation @ rodrigues(u * (0.5 * ds))
                position = position + half_rotation @ np.array([0.0, 0.0, ds], dtype=float)
                rotation = rotation @ rodrigues(u * ds)
                current_s += ds
                positions.append(position.copy())
                rotations.append(rotation.copy())
                u_values.append(u.copy())
                v_values.append(np.array([0.0, 0.0, 1.0], dtype=float))
                s_values.append(current_s)
                segment_ids.append(segment_index)

        return TDCRShape(
            s=np.asarray(s_values, dtype=float),
            p=np.asarray(positions, dtype=float),
            R=np.asarray(rotations, dtype=float),
            u=np.asarray(u_values, dtype=float),
            v=np.asarray(v_values, dtype=float),
            segment_id=np.asarray(segment_ids, dtype=int),
            tau=np.asarray(tau, dtype=float).copy(),
            diagnostics={"solver": "pcca_contact_3d"},
        )

