from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

from .base import TDCRSolverState
from .pcc_3d import PCC3DSolver


@dataclass
class TDCRModelingOracleSolver:
    robot_params: TDCRRobotParams
    modelling_approach: str = "VC_Ref"
    fallback_to_pcc: bool = True

    def __post_init__(self) -> None:
        self._fallback_solver = PCC3DSolver(self.robot_params)
        self._available = False
        self._availability_reason = "pytdcrsv_not_checked"
        try:
            import pytdcrsv  # noqa: F401

            self._available = True
            self._availability_reason = "pytdcrsv_available_but_3_segment_adapter_not_implemented"
        except Exception as exc:
            self._available = False
            self._availability_reason = f"pytdcrsv_unavailable: {exc}"

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        if self.robot_params.num_segments != 3 or self.robot_params.tendons_per_segment != 3:
            raise ValueError("This project oracle adapter expects a 3-segment, 3-tendon-per-segment TDCR.")
        if not self.fallback_to_pcc:
            raise NotImplementedError(
                "The bundled tdcr-modeling/pytdcrsv code is primarily a 2-segment implementation. "
                "Extend TendonDrivenRobot to 3 segments and 9 tendons before using it as a strict oracle."
            )
        shape = self._fallback_solver.solve(tau=tau, obstacles=obstacles, state=state)
        shape.diagnostics = {
            **dict(shape.diagnostics),
            "solver": "tdcr_modeling_oracle",
            "status": "fallback_to_pcc",
            "reason": self._availability_reason,
            "required_extension": "generalize tdcr-modeling fixed 2-segment/6-tendon arrays to 3 segments/9 tendons",
        }
        return shape

