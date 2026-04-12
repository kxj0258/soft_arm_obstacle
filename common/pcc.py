from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .geometry import angle_of, normalize, rotation_from_direction


@dataclass
class PCCSegment2D:
    length: float
    kappa: float
    start_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    start_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))
    lock_start_direction: bool = False
    end_position: np.ndarray = field(init=False)
    end_direction: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.length = float(self.length)
        self.kappa = float(self.kappa)
        if self.length < 0.0:
            raise ValueError("Segment length must be non-negative.")
        self.start_position = np.asarray(self.start_position, dtype=float).copy()
        self.start_direction = normalize(self.start_direction)
        self.end_position = self.start_position.copy()
        self.end_direction = self.start_direction.copy()
        self.update_geometry()

    def copy(self) -> "PCCSegment2D":
        return PCCSegment2D(
            length=self.length,
            kappa=self.kappa,
            start_position=self.start_position.copy(),
            start_direction=self.start_direction.copy(),
            lock_start_direction=self.lock_start_direction,
        )

    def update_start(self, start_position: np.ndarray, start_direction: np.ndarray) -> None:
        self.start_position = np.asarray(start_position, dtype=float).copy()
        self.start_direction = normalize(start_direction)
        self.update_geometry()

    def pose_at_fraction(self, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        arclength = self.length * fraction
        rotation = rotation_from_direction(self.start_direction)

        if abs(self.kappa) < 1e-8:
            local_position = np.array([arclength, 0.0], dtype=float)
            local_direction = np.array([1.0, 0.0], dtype=float)
        else:
            theta = self.kappa * arclength
            local_position = np.array(
                [
                    np.sin(theta) / self.kappa,
                    (1.0 - np.cos(theta)) / self.kappa,
                ],
                dtype=float,
            )
            local_direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)

        world_position = self.start_position + rotation @ local_position
        world_direction = normalize(rotation @ local_direction)
        return world_position, world_direction

    def point_at(self, fraction: float) -> np.ndarray:
        return self.pose_at_fraction(fraction)[0]

    def tangent_at(self, fraction: float) -> np.ndarray:
        return self.pose_at_fraction(fraction)[1]

    def update_geometry(self) -> None:
        self.end_position, self.end_direction = self.pose_at_fraction(1.0)

    def sample_points(self, num_samples: int | None = None) -> np.ndarray:
        if num_samples is None:
            num_samples = max(8, int(np.ceil(self.length * 2.0)) + 1)
        fractions = np.linspace(0.0, 1.0, num_samples)
        return np.stack([self.point_at(fraction) for fraction in fractions], axis=0)

    @property
    def end_angle(self) -> float:
        return angle_of(self.end_direction)


class ContinuumArm2D:
    def __init__(self, segments: Iterable[PCCSegment2D]):
        self.segments = list(segments)

    def update_chain(
        self,
        start_position: np.ndarray | None = None,
        start_direction: np.ndarray | None = None,
    ) -> None:
        if not self.segments:
            return

        if start_position is None:
            current_position = self.segments[0].start_position
        else:
            current_position = np.asarray(start_position, dtype=float)

        if start_direction is None:
            current_direction = self.segments[0].start_direction
        else:
            current_direction = normalize(start_direction)

        for index, segment in enumerate(self.segments):
            if index == 0:
                if segment.lock_start_direction and start_direction is None:
                    current_direction = segment.start_direction
                segment.update_start(current_position, current_direction)
                continue

            previous = self.segments[index - 1]
            current_position = previous.end_position
            current_direction = (
                segment.start_direction if segment.lock_start_direction else previous.end_direction
            )
            segment.update_start(current_position, current_direction)

    @property
    def tip_position(self) -> np.ndarray:
        if not self.segments:
            return np.zeros(2, dtype=float)
        return self.segments[-1].end_position

    @property
    def tip_direction(self) -> np.ndarray:
        if not self.segments:
            return np.array([1.0, 0.0], dtype=float)
        return self.segments[-1].end_direction

    @property
    def tip_angle(self) -> float:
        return angle_of(self.tip_direction)

    def sample_points(self, points_per_segment: int = 32) -> np.ndarray:
        if not self.segments:
            return np.zeros((0, 2), dtype=float)

        samples: list[np.ndarray] = []
        for index, segment in enumerate(self.segments):
            points = segment.sample_points(max(3, points_per_segment))
            if index:
                points = points[1:]
            samples.append(points)
        return np.concatenate(samples, axis=0)


def clone_segments(segments: Iterable[PCCSegment2D]) -> list[PCCSegment2D]:
    return [segment.copy() for segment in segments]
