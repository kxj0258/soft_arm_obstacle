from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import TDCRShape
from tdcr.solvers import (
    CosseratStaticSolver,
    HybridTDCRSolver,
    PCC3DSolver,
    PCCAContact3DSolver,
    PINNTDCRSolver,
    TDCRModelingOracleSolver,
    TDCRSolverState,
)

from .physics_3d import Obstacle3D, ensure_obstacles, make_empty_contact_summary


@dataclass(frozen=True)
class ArmModel3DConfig:
    robot_params: TDCRRobotParams = field(default_factory=TDCRRobotParams)
    solver_backend: str = "hybrid"
    cosserat_modelling_approach: str = "VC_Ref"
    cosserat_number_disks: tuple[int, int] = (8, 8)
    cosserat_gxf_tol: float = 1e-6
    cosserat_step_size: float = 1e-3
    pinn_checkpoint_path: str | None = None
    pinn_device: str = "cpu"
    tau_initial: float = 0.0
    contact_activation_clearance: float = 0.003
    pcca_clearance_margin: float = 1e-4
    pcca_max_iterations: int = 80


def _make_solver(config: ArmModel3DConfig):
    params = config.robot_params
    backend = str(config.solver_backend)
    if backend == "pcc3d":
        return PCC3DSolver(params)
    if backend == "cosserat_static":
        return CosseratStaticSolver(
            params,
            modelling_approach=str(config.cosserat_modelling_approach),
            number_disks=tuple(int(v) for v in config.cosserat_number_disks),
            gxf_tol=float(config.cosserat_gxf_tol),
            step_size=float(config.cosserat_step_size),
        )
    if backend == "pcca_contact_3d":
        return PCCAContact3DSolver(
            params,
            clearance_margin=float(config.pcca_clearance_margin),
            max_iterations=int(config.pcca_max_iterations),
        )
    if backend == "pinn":
        return PINNTDCRSolver(
            params,
            checkpoint_path=config.pinn_checkpoint_path,
            device=config.pinn_device,
        )
    if backend == "hybrid":
        pinn_solver = PINNTDCRSolver(
            params,
            checkpoint_path=config.pinn_checkpoint_path,
            device=config.pinn_device,
        )
        contact_solver = PCCAContact3DSolver(
            params,
            clearance_margin=float(config.pcca_clearance_margin),
            max_iterations=int(config.pcca_max_iterations),
        )
        return HybridTDCRSolver(
            params,
            pinn_solver=pinn_solver,
            contact_solver=contact_solver,
            contact_activation_clearance=float(config.contact_activation_clearance),
        )
    if backend == "oracle":
        return TDCRModelingOracleSolver(params)
    raise ValueError(
        "solver_backend must be one of 'pcc3d', 'cosserat_static', 'pcca_contact_3d', "
        "'pinn', 'hybrid', or 'oracle'."
    )


class ArmModel3D:
    def __init__(
        self,
        config: ArmModel3DConfig | None = None,
        obstacles: list[Obstacle3D | dict | tuple | list] | None = None,
        tau: np.ndarray | None = None,
    ):
        self.config = config or ArmModel3DConfig()
        self.robot_params = self.config.robot_params
        self.solver = _make_solver(self.config)
        self.tau = self._sanitize_tau(tau)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.state = TDCRSolverState(tau=self.tau.copy())
        self.shape: TDCRShape | None = None
        self.contact_backend_used = self.config.solver_backend
        self.contact_backend_details: dict[str, object] = {
            "requested_backend": self.config.solver_backend,
            "used_backend": self.config.solver_backend,
            "status": "initialized",
            "reason": "model_initialized",
        }
        self.rebuild()

    @property
    def num_segments(self) -> int:
        return self.robot_params.num_segments

    @property
    def tendons_per_segment(self) -> int:
        return self.robot_params.tendons_per_segment

    def _default_tau(self) -> np.ndarray:
        return np.full(
            (self.num_segments, self.tendons_per_segment),
            float(self.config.tau_initial),
            dtype=float,
        )

    def _sanitize_tau(self, tau: np.ndarray | None) -> np.ndarray:
        values = self._default_tau() if tau is None else np.asarray(tau, dtype=float)
        return self.robot_params.validate_tau(values)

    def rebuild(self) -> None:
        self.state.tau = self.tau.copy()
        shape = self.solver.solve(tau=self.tau, obstacles=self.obstacles, state=self.state)
        self.shape = shape
        self.state.last_shape = shape.copy()
        self.state.diagnostics = dict(shape.diagnostics)
        self.contact_backend_used = str(shape.diagnostics.get("solver", self.config.solver_backend))
        self.contact_backend_details = {
            "requested_backend": self.config.solver_backend,
            "used_backend": self.contact_backend_used,
            "status": shape.diagnostics.get("status", "active"),
            "reason": shape.diagnostics.get("reason", shape.diagnostics.get("contact_state", "")),
            "contact_state": shape.contact_summary.contact_state,
            **dict(shape.diagnostics),
        }

    def reset(
        self,
        *,
        obstacles: list[Obstacle3D | dict | tuple | list] | None = None,
        tau: np.ndarray | None = None,
    ) -> None:
        self.tau = self._sanitize_tau(tau)
        self.obstacles = ensure_obstacles(obstacles or [])
        self.state = TDCRSolverState(tau=self.tau.copy())
        self.rebuild()

    def snapshot(self) -> dict[str, object]:
        return {
            "tau": self.tau.copy(),
            "obstacles": ensure_obstacles(self.obstacles),
            "solver_state": self.state.snapshot(),
        }

    def restore(self, state: dict[str, object]) -> None:
        self.tau = self._sanitize_tau(np.asarray(state["tau"], dtype=float))
        self.obstacles = ensure_obstacles(state.get("obstacles", []))
        solver_state = state.get("solver_state")
        self.state = (
            TDCRSolverState(tau=self.tau.copy())
            if solver_state is None
            else TDCRSolverState.restore(solver_state)
        )
        self.rebuild()

    def set_obstacles(self, obstacles: list[Obstacle3D | dict | tuple | list]) -> None:
        self.obstacles = ensure_obstacles(obstacles)
        self.rebuild()

    def apply_action(self, tau: np.ndarray) -> None:
        self.tau = self._sanitize_tau(tau)
        self.rebuild()

    def strain_u_by_segment(self) -> np.ndarray:
        shape = self._shape()
        values = np.zeros((self.num_segments, 3), dtype=float)
        for segment_index in range(self.num_segments):
            mask = shape.segment_id == segment_index
            if np.any(mask):
                values[segment_index] = np.mean(shape.u[mask], axis=0)
        return values

    def _shape(self) -> TDCRShape:
        if self.shape is None:
            raise RuntimeError("3D arm shape has not been built.")
        return self.shape

    @property
    def root_position(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    @property
    def tip_position(self) -> np.ndarray:
        return self._shape().tip_position

    @property
    def tip_rotation(self) -> np.ndarray:
        return self._shape().tip_rotation

    @property
    def tip_rotvec(self) -> np.ndarray:
        return self._shape().tip_rotvec

    @property
    def contact_summary(self):
        return self._shape().contact_summary if self.shape is not None else make_empty_contact_summary(self.num_segments)

    @property
    def collision_events(self) -> list[np.ndarray]:
        summary = self.contact_summary
        if summary.active:
            return [summary.nearest_point.copy()]
        return []

    def tip_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.tip_position.astype(np.float32),
                self.tip_rotvec.astype(np.float32),
            )
        ).astype(np.float32)

    def free_polyline(self) -> np.ndarray:
        return self._shape().p.copy()

    def observed_polyline(self) -> np.ndarray:
        return self._shape().p.copy()

    def backbone_points(self, num_points: int, *, include_root: bool = False) -> np.ndarray:
        return self._shape().sample_keypoints(num_points, include_root=include_root)

    def contact_summary_dict(self) -> dict:
        return self.contact_summary.as_dict()
