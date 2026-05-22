from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.frames import hat, normalize, rodrigues
from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import ContactSummary3D, Obstacle3D, TDCRShape

from .base import TDCRSolverState


@dataclass
class PCC3DSolver:
    robot_params: TDCRRobotParams
    contact_stiffness: float = 80.0

    def solve(
        self,
        *,
        tau: np.ndarray,
        obstacles: list[Obstacle3D] | None = None,
        state: TDCRSolverState | None = None,
        contacts_hint: np.ndarray | None = None,
    ) -> TDCRShape:
        tau = self.robot_params.validate_tau(tau)
        shape = self._integrate(tau=tau)
        summary = summarize_shape_contacts(
            shape=shape,
            obstacles=obstacles or [],
            arm_radius=self.robot_params.arm_radius,
            num_segments=self.robot_params.num_segments,
            contact_stiffness=self.contact_stiffness,
        )
        shape.contact_summary = summary
        shape.diagnostics = {
            **dict(shape.diagnostics),
            "solver": "pcc3d",
            "contact_state": summary.contact_state,
        }
        return shape

    def _curvature_from_tau(self, tau: np.ndarray) -> np.ndarray:
        routing = self.robot_params.tendon_routing_vectors()
        centered_tau = tau - np.mean(tau, axis=1, keepdims=True)
        moments = np.zeros((self.robot_params.num_segments, 3), dtype=float)
        for segment_index in range(self.robot_params.num_segments):
            for tendon_index in range(self.robot_params.tendons_per_segment):
                force = np.array([0.0, 0.0, -centered_tau[segment_index, tendon_index]], dtype=float)
                moments[segment_index] += np.cross(routing[tendon_index], force)
        scale = self.robot_params.curvature_gain / max(self.robot_params.tendon_routing_radius, 1e-8)
        curvature = scale * moments
        curvature[:, 2] += self.robot_params.torsion_gain * (
            centered_tau[:, 0] - centered_tau[:, 1] + 0.5 * centered_tau[:, 2]
        )
        limit = float(self.robot_params.curvature_limit)
        norms = np.linalg.norm(curvature, axis=1)
        too_large = norms > limit
        if np.any(too_large):
            curvature[too_large] *= (limit / np.maximum(norms[too_large], 1e-8))[:, None]
        return curvature

    def _integrate(self, *, tau: np.ndarray) -> TDCRShape:
        params = self.robot_params
        curvature = self._curvature_from_tau(tau)
        positions: list[np.ndarray] = [np.zeros(3, dtype=float)]
        rotations: list[np.ndarray] = [np.eye(3, dtype=float)]
        strains_u: list[np.ndarray] = [curvature[0].copy()]
        strains_v: list[np.ndarray] = [np.array([0.0, 0.0, 1.0], dtype=float)]
        arclengths: list[float] = [0.0]
        segment_ids: list[int] = [0]

        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        current_s = 0.0
        for segment_index, length in enumerate(params.segment_lengths):
            steps = max(int(params.samples_per_segment), 2)
            ds = float(length) / float(steps)
            u = curvature[segment_index]
            for _ in range(steps):
                # Midpoint-style update gives stable enough geometry for the RL baseline.
                half_rotation = rotation @ rodrigues(u * (0.5 * ds))
                position = position + half_rotation @ np.array([0.0, 0.0, ds], dtype=float)
                rotation = rotation @ rodrigues(u * ds)
                current_s += ds
                positions.append(position.copy())
                rotations.append(rotation.copy())
                strains_u.append(u.copy())
                strains_v.append(np.array([0.0, 0.0, 1.0], dtype=float))
                arclengths.append(current_s)
                segment_ids.append(segment_index)

        return TDCRShape(
            s=np.asarray(arclengths, dtype=float),
            p=np.asarray(positions, dtype=float),
            R=np.asarray(rotations, dtype=float),
            u=np.asarray(strains_u, dtype=float),
            v=np.asarray(strains_v, dtype=float),
            segment_id=np.asarray(segment_ids, dtype=int),
            tau=tau.copy(),
            diagnostics={
                "solver": "pcc3d",
                "curvature": curvature.copy(),
            },
        )


def summarize_shape_contacts(
    *,
    shape: TDCRShape,
    obstacles: list[Obstacle3D],
    arm_radius: float,
    num_segments: int,
    contact_stiffness: float = 80.0,
    activation_clearance: float = 0.002,
) -> ContactSummary3D:
    segment_clearances = np.full(int(num_segments), float("inf"), dtype=float)
    segment_contact_mask = np.zeros(int(num_segments), dtype=bool)
    if not obstacles or shape.p.size == 0:
        return ContactSummary3D(
            segment_clearances=segment_clearances,
            segment_contact_mask=segment_contact_mask,
            contact_occurrence_rate=0.0,
            micro_step_count=1,
        )

    best: tuple[float, int, int] | None = None
    collision_count = 0
    for obstacle_index, obstacle in enumerate(obstacles):
        clearances = obstacle.clearance(shape.p, inflation=float(arm_radius))
        collision_count += int(np.sum(clearances < 0.0))
        for segment_index in range(int(num_segments)):
            mask = shape.segment_id == segment_index
            if np.any(mask):
                segment_clearances[segment_index] = min(
                    float(segment_clearances[segment_index]),
                    float(np.min(clearances[mask])),
                )
        local_index = int(np.argmin(clearances))
        local_clearance = float(clearances[local_index])
        if best is None or local_clearance < best[0]:
            best = (local_clearance, obstacle_index, local_index)

    assert best is not None
    min_clearance, obstacle_index, sample_index = best
    obstacle = obstacles[obstacle_index]
    point = shape.p[sample_index].copy()
    normal = normalize(obstacle.normal_at(point), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    penetration = max(0.0, -float(min_clearance))
    force = normal * penetration * float(contact_stiffness)
    active = min_clearance <= float(activation_clearance)
    segment_contact_mask = segment_clearances <= float(activation_clearance)
    state = "collision" if penetration > 0.0 else ("near" if active else "free")
    return ContactSummary3D(
        active=bool(active),
        contact_state=state,
        collision_count=int(collision_count),
        min_clearance=float(min_clearance),
        nearest_obstacle_index=int(obstacle_index),
        nearest_sample_index=int(sample_index),
        nearest_segment_index=int(shape.segment_id[sample_index]),
        nearest_s=float(shape.s[sample_index]),
        nearest_point=point,
        nearest_obstacle_center=obstacle.center.copy(),
        nearest_obstacle_radii=obstacle.radii.copy(),
        contact_normal=normal,
        contact_force=force,
        primary_penetration=float(penetration),
        segment_clearances=segment_clearances,
        segment_contact_mask=segment_contact_mask,
        contact_occurrence_rate=float(active),
        micro_step_count=1,
    )
