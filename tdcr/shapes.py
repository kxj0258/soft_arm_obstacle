from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import rotvec_from_matrix


@dataclass(frozen=True)
class Obstacle3D:
    center: np.ndarray
    radii: np.ndarray | float
    kind: str = "sphere"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3).copy())
        radii = np.asarray(self.radii, dtype=float)
        if radii.ndim == 0:
            radii = np.repeat(float(radii), 3)
        radii = radii.reshape(3).copy()
        if np.any(radii <= 0.0):
            raise ValueError("Obstacle radii must be positive.")
        object.__setattr__(self, "radii", radii)

    @property
    def radius(self) -> float:
        return float(np.max(self.radii))

    def clearance(self, points: np.ndarray, *, inflation: float = 0.0) -> np.ndarray:
        samples = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.kind == "sphere" and np.allclose(self.radii, self.radii[0]):
            values = np.linalg.norm(samples - self.center[None, :], axis=1) - (float(self.radii[0]) + inflation)
            return values.astype(float)
        scaled = (samples - self.center[None, :]) / np.maximum(self.radii[None, :] + inflation, 1e-9)
        norm = np.linalg.norm(scaled, axis=1)
        return (norm - 1.0) * float(np.mean(self.radii))

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(3)
        if self.kind == "sphere" and np.allclose(self.radii, self.radii[0]):
            direction = point - self.center
        else:
            direction = (point - self.center) / np.maximum(self.radii ** 2, 1e-12)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return direction / norm

    def as_slot(self) -> np.ndarray:
        return np.array(
            [
                1.0,
                0.0 if self.kind == "sphere" else 1.0,
                *self.center.tolist(),
                *np.asarray(self.radii, dtype=float).tolist(),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )


@dataclass
class ContactSummary3D:
    active: bool = False
    contact_state: str = "free"
    collision_count: int = 0
    min_clearance: float = float("inf")
    nearest_obstacle_index: int = -1
    nearest_sample_index: int = -1
    nearest_segment_index: int = -1
    nearest_s: float = 0.0
    nearest_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    nearest_obstacle_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    nearest_obstacle_radii: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    contact_normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    primary_penetration: float = 0.0
    segment_clearances: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    segment_contact_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    contact_occurrence_rate: float = 0.0
    micro_step_count: int = 1

    def copy(self) -> "ContactSummary3D":
        return ContactSummary3D(
            active=bool(self.active),
            contact_state=str(self.contact_state),
            collision_count=int(self.collision_count),
            min_clearance=float(self.min_clearance),
            nearest_obstacle_index=int(self.nearest_obstacle_index),
            nearest_sample_index=int(self.nearest_sample_index),
            nearest_segment_index=int(self.nearest_segment_index),
            nearest_s=float(self.nearest_s),
            nearest_point=self.nearest_point.copy(),
            nearest_obstacle_center=self.nearest_obstacle_center.copy(),
            nearest_obstacle_radii=self.nearest_obstacle_radii.copy(),
            contact_normal=self.contact_normal.copy(),
            contact_force=self.contact_force.copy(),
            primary_penetration=float(self.primary_penetration),
            segment_clearances=self.segment_clearances.copy(),
            segment_contact_mask=self.segment_contact_mask.copy(),
            contact_occurrence_rate=float(self.contact_occurrence_rate),
            micro_step_count=int(self.micro_step_count),
        )

    def as_dict(self) -> dict:
        return {
            "active": bool(self.active),
            "contact_state": self.contact_state,
            "collision_count": int(self.collision_count),
            "min_clearance": float(self.min_clearance),
            "nearest_obstacle_index": int(self.nearest_obstacle_index),
            "nearest_sample_index": int(self.nearest_sample_index),
            "nearest_segment_index": int(self.nearest_segment_index),
            "nearest_s": float(self.nearest_s),
            "nearest_point": self.nearest_point.copy(),
            "nearest_obstacle_center": self.nearest_obstacle_center.copy(),
            "nearest_obstacle_radii": self.nearest_obstacle_radii.copy(),
            "contact_normal": self.contact_normal.copy(),
            "contact_force": self.contact_force.copy(),
            "primary_penetration": float(self.primary_penetration),
            "segment_clearances": self.segment_clearances.copy(),
            "segment_contact_mask": self.segment_contact_mask.copy(),
            "contact_occurrence_rate": float(self.contact_occurrence_rate),
            "micro_step_count": int(self.micro_step_count),
        }

    def contact_slots(self, *, max_contacts: int = 4, total_length: float = 1.0) -> np.ndarray:
        slots = np.zeros((int(max_contacts), 24), dtype=np.float32)
        if not self.active:
            return slots
        slots[0] = np.array(
            [
                1.0,
                float(max(self.nearest_segment_index, 0)),
                float(self.nearest_s) / max(float(total_length), 1e-8),
                *self.nearest_point.tolist(),
                *self.contact_normal.tolist(),
                *self.contact_force.tolist(),
                float(max(self.nearest_obstacle_index, 0)),
                *self.nearest_obstacle_center.tolist(),
                *self.nearest_obstacle_radii.tolist(),
                float(self.min_clearance),
                float(self.primary_penetration),
                0.0,
                0.0 if self.contact_state == "free" else 1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return slots


@dataclass
class TDCRShape:
    s: np.ndarray
    p: np.ndarray
    R: np.ndarray
    u: np.ndarray
    v: np.ndarray
    segment_id: np.ndarray
    tau: np.ndarray
    contact_summary: ContactSummary3D = field(default_factory=ContactSummary3D)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.s = np.asarray(self.s, dtype=float).reshape(-1)
        count = self.s.size
        self.p = np.asarray(self.p, dtype=float).reshape(count, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(count, 3, 3)
        self.u = np.asarray(self.u, dtype=float).reshape(count, 3)
        self.v = np.asarray(self.v, dtype=float).reshape(count, 3)
        self.segment_id = np.asarray(self.segment_id, dtype=int).reshape(count)
        self.tau = np.asarray(self.tau, dtype=float)

    @property
    def tip_position(self) -> np.ndarray:
        return self.p[-1].copy()

    @property
    def tip_rotation(self) -> np.ndarray:
        return self.R[-1].copy()

    @property
    def tip_rotvec(self) -> np.ndarray:
        return rotvec_from_matrix(self.R[-1])

    @property
    def total_length(self) -> float:
        return float(self.s[-1]) if self.s.size else 0.0

    def copy(self) -> "TDCRShape":
        return TDCRShape(
            s=self.s.copy(),
            p=self.p.copy(),
            R=self.R.copy(),
            u=self.u.copy(),
            v=self.v.copy(),
            segment_id=self.segment_id.copy(),
            tau=self.tau.copy(),
            contact_summary=self.contact_summary.copy(),
            diagnostics=dict(self.diagnostics),
        )

    def sample_keypoints(self, num_points: int, *, include_root: bool = False) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0, 3), dtype=float)
        if self.s.size == 0:
            return np.zeros((num_points, 3), dtype=float)
        start = 0.0 if include_root else 1.0 / float(num_points)
        fractions = np.linspace(start, 1.0, num_points)
        query_s = fractions * max(self.total_length, 1e-8)
        values = np.column_stack(
            [
                np.interp(query_s, self.s, self.p[:, axis])
                for axis in range(3)
            ]
        )
        return values.astype(float)
