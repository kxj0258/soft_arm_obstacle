from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

from .base import TDCRSolverState
from .pcc_3d import PCC3DSolver


@dataclass
class PINNTDCRSolver:
    robot_params: TDCRRobotParams
    checkpoint_path: str | None = None
    device: str = "cpu"
    fallback_to_pcc: bool = True

    def __post_init__(self) -> None:
        self._model = None
        self._fallback_solver = PCC3DSolver(self.robot_params)
        if self.checkpoint_path is not None:
            self._try_load_checkpoint(self.checkpoint_path)

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        if self._model is None:
            if not self.fallback_to_pcc:
                raise RuntimeError("PINN checkpoint is not loaded.")
            shape = self._fallback_solver.solve(tau=tau, obstacles=obstacles, state=state)
            shape.diagnostics = {
                **dict(shape.diagnostics),
                "solver": "tdcr3d_pinn",
                "status": "fallback_to_pcc",
                "reason": "checkpoint_not_loaded",
            }
            return shape

        # A trained project-native TDCR3DPINN can be plugged in here. The fallback
        # path keeps the environment runnable before training data exists.
        shape = self._fallback_solver.solve(tau=tau, obstacles=obstacles, state=state)
        shape.diagnostics = {
            **dict(shape.diagnostics),
            "solver": "tdcr3d_pinn",
            "status": "model_loaded_fallback_geometry",
        }
        return shape

    def _try_load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            if self.fallback_to_pcc:
                self._model = None
                return
            raise FileNotFoundError(path)
        try:
            import torch

            payload = torch.load(path, map_location=self.device)
            self._model = payload
        except Exception:
            if not self.fallback_to_pcc:
                raise
            self._model = None

