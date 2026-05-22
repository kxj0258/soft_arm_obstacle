from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D, TDCRShape

CONTACT_SLOT_DIM = 24
OBSTACLE_SLOT_DIM = 10
DEFAULT_MAX_CONTACTS = 4
DEFAULT_MAX_OBSTACLES = 8


@dataclass
class TDCRSample:
    """Contact-aware TDCR sample used by the oracle/PINN pipeline.

    Shapes:
    - ``tau``: ``(num_segments, tendons_per_segment)``
    - ``s``: ``(num_points,)``
    - ``p``: ``(num_points, 3)``
    - ``R``: ``(num_points, 3, 3)``
    - ``u`` / ``v``: ``(num_points, 3)``
    - ``segment_id``: ``(num_points,)``
    - ``contact_slots``: ``(max_contacts, 24)``
    - ``obstacle_slots``: ``(max_obstacles, 10)``
    """

    tau: np.ndarray
    s: np.ndarray
    p: np.ndarray
    R: np.ndarray
    u: np.ndarray
    v: np.ndarray
    segment_id: np.ndarray
    contact_slots: np.ndarray
    obstacle_slots: np.ndarray
    robot_params: np.ndarray
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.tau = np.asarray(self.tau, dtype=np.float32)
        self.s = np.asarray(self.s, dtype=np.float32).reshape(-1)
        count = self.s.size
        self.p = np.asarray(self.p, dtype=np.float32).reshape(count, 3)
        self.R = np.asarray(self.R, dtype=np.float32).reshape(count, 3, 3)
        self.u = np.asarray(self.u, dtype=np.float32).reshape(count, 3)
        self.v = np.asarray(self.v, dtype=np.float32).reshape(count, 3)
        self.segment_id = np.asarray(self.segment_id, dtype=np.int64).reshape(count)
        self.contact_slots = np.asarray(self.contact_slots, dtype=np.float32).reshape(-1, CONTACT_SLOT_DIM)
        self.obstacle_slots = np.asarray(self.obstacle_slots, dtype=np.float32).reshape(-1, OBSTACLE_SLOT_DIM)
        self.robot_params = np.asarray(self.robot_params, dtype=np.float32).reshape(-1)
        self.metadata = dict(self.metadata or {})

    @property
    def tip_position(self) -> np.ndarray:
        return self.p[-1].copy()

    def as_npz_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}tau": self.tau,
            f"{prefix}s": self.s,
            f"{prefix}p": self.p,
            f"{prefix}R": self.R,
            f"{prefix}u": self.u,
            f"{prefix}v": self.v,
            f"{prefix}segment_id": self.segment_id,
            f"{prefix}contact_slots": self.contact_slots,
            f"{prefix}obstacle_slots": self.obstacle_slots,
            f"{prefix}robot_params": self.robot_params,
        }


@dataclass
class TDCRDatasetBatch:
    tau: np.ndarray
    s: np.ndarray
    p: np.ndarray
    R: np.ndarray
    u: np.ndarray
    v: np.ndarray
    segment_id: np.ndarray
    contact_slots: np.ndarray
    obstacle_slots: np.ndarray
    robot_params: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "tau": self.tau,
            "s": self.s,
            "p": self.p,
            "R": self.R,
            "u": self.u,
            "v": self.v,
            "segment_id": self.segment_id,
            "contact_slots": self.contact_slots,
            "obstacle_slots": self.obstacle_slots,
            "robot_params": self.robot_params,
        }


def obstacle_to_slot(obstacle: Obstacle3D) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            0.0 if obstacle.kind == "sphere" else 1.0,
            *obstacle.center.astype(float).tolist(),
            *obstacle.radii.astype(float).tolist(),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def obstacles_to_slots(
    obstacles: list[Obstacle3D] | tuple[Obstacle3D, ...],
    *,
    max_obstacles: int = DEFAULT_MAX_OBSTACLES,
) -> np.ndarray:
    slots = np.zeros((int(max_obstacles), OBSTACLE_SLOT_DIM), dtype=np.float32)
    for index, obstacle in enumerate(list(obstacles)[: int(max_obstacles)]):
        slots[index] = obstacle_to_slot(obstacle)
    return slots


def sample_from_shape(
    shape: TDCRShape,
    *,
    robot_params: TDCRRobotParams,
    obstacles: list[Obstacle3D] | None = None,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    max_obstacles: int = DEFAULT_MAX_OBSTACLES,
    metadata: dict | None = None,
) -> TDCRSample:
    return TDCRSample(
        tau=shape.tau.astype(np.float32),
        s=shape.s.astype(np.float32),
        p=shape.p.astype(np.float32),
        R=shape.R.astype(np.float32),
        u=shape.u.astype(np.float32),
        v=shape.v.astype(np.float32),
        segment_id=shape.segment_id.astype(np.int64),
        contact_slots=shape.contact_summary.contact_slots(
            max_contacts=max_contacts,
            total_length=robot_params.total_length,
        ),
        obstacle_slots=obstacles_to_slots(obstacles or [], max_obstacles=max_obstacles),
        robot_params=robot_params.normalized_params(),
        metadata=metadata,
    )


def collate_samples(samples: list[TDCRSample] | tuple[TDCRSample, ...]) -> TDCRDatasetBatch:
    if not samples:
        raise ValueError("At least one TDCRSample is required.")
    first_count = samples[0].s.size
    for index, sample in enumerate(samples):
        if sample.s.size != first_count:
            raise ValueError(
                "All samples in a batch must use the same arclength grid. "
                f"Sample 0 has {first_count} points but sample {index} has {sample.s.size}."
            )
    return TDCRDatasetBatch(
        tau=np.stack([sample.tau for sample in samples], axis=0).astype(np.float32),
        s=np.stack([sample.s for sample in samples], axis=0).astype(np.float32),
        p=np.stack([sample.p for sample in samples], axis=0).astype(np.float32),
        R=np.stack([sample.R for sample in samples], axis=0).astype(np.float32),
        u=np.stack([sample.u for sample in samples], axis=0).astype(np.float32),
        v=np.stack([sample.v for sample in samples], axis=0).astype(np.float32),
        segment_id=np.stack([sample.segment_id for sample in samples], axis=0).astype(np.int64),
        contact_slots=np.stack([sample.contact_slots for sample in samples], axis=0).astype(np.float32),
        obstacle_slots=np.stack([sample.obstacle_slots for sample in samples], axis=0).astype(np.float32),
        robot_params=np.stack([sample.robot_params for sample in samples], axis=0).astype(np.float32),
    )
