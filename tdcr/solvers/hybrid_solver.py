from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

from .base import TDCRSolverState
from .pcca_contact_3d import PCCAContact3DSolver
from .pinn_solver import PINNTDCRSolver


@dataclass
class HybridTDCRSolver:
    robot_params: TDCRRobotParams
    pinn_solver: PINNTDCRSolver | None = None
    contact_solver: PCCAContact3DSolver | None = None
    contact_activation_clearance: float = 0.003

    def __post_init__(self) -> None:
        if self.pinn_solver is None:
            self.pinn_solver = PINNTDCRSolver(self.robot_params)
        if self.contact_solver is None:
            self.contact_solver = PCCAContact3DSolver(self.robot_params)

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        obstacles = obstacles or []
        assert self.pinn_solver is not None
        assert self.contact_solver is not None
        free_shape = self.pinn_solver.solve(
            tau=tau,
            obstacles=obstacles,
            state=state,
            contacts_hint=contacts_hint,
        )
        if not obstacles or free_shape.contact_summary.min_clearance > self.contact_activation_clearance:
            free_shape.diagnostics = {
                **dict(free_shape.diagnostics),
                "solver": "hybrid_pinn_contact",
                "status": "free_shape_accepted",
            }
            return free_shape

        shape = self.contact_solver.solve(
            tau=tau,
            obstacles=obstacles,
            state=state,
            contacts_hint=free_shape.contact_summary.contact_slots(total_length=self.robot_params.total_length),
        )
        shape.diagnostics = {
            **dict(shape.diagnostics),
            "solver": "hybrid_pinn_contact",
            "status": "contact_corrected",
            "free_min_clearance": float(free_shape.contact_summary.min_clearance),
        }
        return shape

