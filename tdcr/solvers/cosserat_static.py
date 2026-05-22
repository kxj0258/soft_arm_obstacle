from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from tdcr.frames import normalize, rotvec_from_matrix
from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

from .base import TDCRSolverState
from .pcc_3d import PCC3DSolver, summarize_shape_contacts


def _maybe_add_pytdcrsv_path() -> None:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "tdcrGit" / "tdcr-pinn" / "Required" / "tdcr-lilge-binding" / "src"
    if candidate.exists():
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _segment_frames_from_disk_frames(
    disk_frames_array: np.ndarray,
    number_disks: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(disk_frames_array, dtype=float)
    if frames.ndim != 2 or frames.shape[0] != 4 or frames.shape[1] % 4 != 0:
        raise ValueError(f"Unexpected disk frame storage shape: {frames.shape!r}")

    total_disks = int(number_disks[0]) + int(number_disks[1])
    expected_cols = (total_disks + 1) * 4
    if frames.shape[1] != expected_cols:
        raise ValueError(
            f"Disk frame storage width {frames.shape[1]} does not match expected {expected_cols} "
            f"for number_disks={number_disks!r}."
        )

    poses = []
    segment_ids = []
    for frame_index in range(total_disks + 1):
        transform = frames[:, 4 * frame_index : 4 * (frame_index + 1)]
        poses.append(transform.copy())
        if frame_index == 0:
            segment_ids.append(0)
        elif frame_index <= int(number_disks[0]):
            segment_ids.append(0)
        else:
            segment_ids.append(1)
    return np.asarray(poses, dtype=float), np.asarray(segment_ids, dtype=int), np.arange(total_disks + 1, dtype=int)


def _full_state_to_uv(full_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(full_state, dtype=float).reshape(-1, 19)
    if values.shape[1] != 19:
        raise ValueError(f"Unexpected full state shape: {values.shape!r}")
    v = values[:, 12:15].copy()
    u = values[:, 15:18].copy()
    return u, v


@dataclass
class CosseratStaticSolver:
    robot_params: TDCRRobotParams
    modelling_approach: str = "VC_Ref"
    number_disks: tuple[int, int] = (8, 8)
    gxf_tol: float = 1e-6
    step_size: float = 1e-3
    fallback_to_pcc: bool = True
    contact_stiffness: float = 120.0

    def __post_init__(self) -> None:
        self._fallback_solver = PCC3DSolver(self.robot_params, contact_stiffness=self.contact_stiffness)
        self._backend = None
        self._availability_reason = "not_checked"
        self._import_error = None
        self._try_import_backend()

    def _try_import_backend(self) -> None:
        try:
            _maybe_add_pytdcrsv_path()
            from pytdcrsv.static_robot_model import StaticRobotModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional local build
            self._backend = None
            self._availability_reason = f"pytdcrsv_unavailable: {exc}"
            self._import_error = exc
            return
        self._backend = StaticRobotModel
        self._availability_reason = "pytdcrsv_available"

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        tau = self.robot_params.validate_tau(tau)
        obstacles = obstacles or []

        try:
            shape = self._solve_with_pytdcrsv(tau=tau, obstacles=obstacles, state=state)
        except Exception as exc:
            if not self.fallback_to_pcc:
                raise
            shape = self._fallback_solver.solve(tau=tau, obstacles=obstacles, state=state)
            shape.diagnostics = {
                **dict(shape.diagnostics),
                "solver": "cosserat_static",
                "status": "fallback_to_pcc",
                "reason": str(exc),
                "availability_reason": self._availability_reason,
                "modelling_approach": self.modelling_approach,
            }
            return shape

        summary = summarize_shape_contacts(
            shape=shape,
            obstacles=obstacles,
            arm_radius=self.robot_params.arm_radius,
            num_segments=self.robot_params.num_segments,
            contact_stiffness=self.contact_stiffness,
        )
        shape.contact_summary = summary
        shape.diagnostics = {
            **dict(shape.diagnostics),
            "solver": "cosserat_static",
            "status": "active",
            "availability_reason": self._availability_reason,
            "modelling_approach": self.modelling_approach,
            "contact_state": summary.contact_state,
        }
        return shape

    def _solve_with_pytdcrsv(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D],
        state: TDCRSolverState | None,
    ) -> TDCRShape:
        if self._backend is None:
            raise RuntimeError(self._availability_reason)
        if self.robot_params.num_segments != 2 or self.robot_params.tendons_per_segment != 3:
            raise RuntimeError(
                "pytdcrsv Cosserat backend currently requires a 2-segment, 3-tendon-per-segment TDCR. "
                f"Received num_segments={self.robot_params.num_segments}, "
                f"tendons_per_segment={self.robot_params.tendons_per_segment}."
            )

        StaticRobotModel = self._backend
        segment_lengths = np.asarray(self.robot_params.segment_lengths, dtype=float)
        pradius_disks = np.full(2, float(self.robot_params.tendon_routing_radius), dtype=float)
        act = np.asarray(tau, dtype=float).reshape(-1, 1)
        robot = StaticRobotModel(
            segment_length__m=segment_lengths,
            f__n=np.zeros(3, dtype=float),
            l__Nm=np.zeros(3, dtype=float),
            youngs_modulus__n_per_m2=float(self.robot_params.youngs_modulus),
            pradius_disks__m=pradius_disks,
            ro__m=float(self.robot_params.backbone_radius),
            modelling_approach=self.modelling_approach,
            number_disks=np.asarray(self.number_disks, dtype=int),
            gxf_tol=float(self.gxf_tol),
            step_size=float(self.step_size),
        )
        robot.use_recomputation = True

        if state is not None and isinstance(state.warm_start, np.ndarray):
            try:
                robot.setDefaultInitialValue(np.asarray(state.warm_start, dtype=float))
                robot.keep_inits_uv = True
            except Exception:
                pass

        success = np.asarray(robot.calc_pose_from_ctrl(act=act), dtype=bool).reshape(-1)
        if success.size != 1 or not bool(success[0]):
            raise RuntimeError(f"Cosserat solve failed with success flags {success.tolist()}.")

        if state is not None:
            try:
                state.warm_start = np.asarray(robot.getFinalInitValues(), dtype=float).copy()
            except Exception:
                state.warm_start = None

        disk_frames = np.asarray(robot.disk_frames_array, dtype=float)
        frames, segment_id, frame_indices = _segment_frames_from_disk_frames(disk_frames, self.number_disks)
        p = frames[:, :3, 3].copy()
        R = frames[:, :3, :3].copy()

        total_disks = int(self.number_disks[0]) + int(self.number_disks[1])
        s = np.zeros(total_disks + 1, dtype=float)
        s[: int(self.number_disks[0]) + 1] = np.linspace(
            0.0,
            float(segment_lengths[0]),
            int(self.number_disks[0]) + 1,
        )
        s[int(self.number_disks[0]) + 1 :] = np.linspace(
            float(segment_lengths[0]) + float(segment_lengths[1]) / max(int(self.number_disks[1]), 1),
            float(np.sum(segment_lengths)),
            int(self.number_disks[1]),
        )

        u = np.zeros((total_disks + 1, 3), dtype=float)
        v = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (total_disks + 1, 1))

        if self.modelling_approach == "VC":
            if robot.full_state1_lst is not None and robot.full_state2_lst is not None:
                seg1_u, seg1_v = _full_state_to_uv(robot.full_state1_lst[0])
                seg2_u, seg2_v = _full_state_to_uv(robot.full_state2_lst[0])
                u[: seg1_u.shape[0]] = seg1_u
                v[: seg1_v.shape[0]] = seg1_v
                u[seg1_u.shape[0] :] = seg2_u
                v[seg1_v.shape[0] :] = seg2_v
        else:
            for index in range(1, total_disks + 1):
                ds = max(float(s[index] - s[index - 1]), 1e-8)
                relative = R[index - 1].T @ R[index]
                u[index] = rotvec_from_matrix(relative) / ds
                v[index] = normalize(R[index - 1].T @ (p[index] - p[index - 1]) / ds)
            u[0] = u[1] if total_disks >= 1 else 0.0

        return TDCRShape(
            s=s,
            p=p,
            R=R,
            u=u,
            v=v,
            segment_id=segment_id,
            tau=tau.copy(),
            diagnostics={
                "solver": "cosserat_static",
                "modelling_approach": self.modelling_approach,
                "number_disks": tuple(int(v) for v in self.number_disks),
                "frame_indices": frame_indices.copy(),
            },
        )
