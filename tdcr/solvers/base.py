from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape


@dataclass
class TDCRSolverState:
    tau: np.ndarray
    warm_start: object | None = None
    last_shape: TDCRShape | None = None
    diagnostics: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "tau": np.asarray(self.tau, dtype=float).copy(),
            "warm_start": None if self.warm_start is None else self._copy_warm_start(self.warm_start),
            "last_shape": None if self.last_shape is None else self.last_shape.copy(),
            "diagnostics": dict(self.diagnostics),
        }

    @staticmethod
    def _copy_warm_start(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return dict(value)
        return value

    @classmethod
    def restore(cls, snapshot: dict) -> "TDCRSolverState":
        return cls(
            tau=np.asarray(snapshot["tau"], dtype=float).copy(),
            warm_start=snapshot.get("warm_start"),
            last_shape=snapshot.get("last_shape"),
            diagnostics=dict(snapshot.get("diagnostics", {})),
        )


class TDCRShapeSolver(Protocol):
    robot_params: TDCRRobotParams

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        ...

