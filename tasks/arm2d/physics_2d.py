from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.geometry import EPSILON, left_normal, normalize
from common.pcc import PCCSegment2D


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).copy()
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError("Obstacle radius must be positive.")


@dataclass
class CollisionEvent:
    obstacle_index: int
    segment_index: int
    fraction: float
    nearest_point: np.ndarray
    contact_point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    penetration: float


def ensure_obstacles(obstacles: Iterable[CircleObstacle | tuple[float, float, float]]) -> list[CircleObstacle]:
    normalized: list[CircleObstacle] = []
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            normalized.append(CircleObstacle(obstacle.center, obstacle.radius))
        else:
            x, y, radius = obstacle
            normalized.append(CircleObstacle(center=np.array([x, y], dtype=float), radius=radius))
    return normalized


def _first_contact_fraction(
    segment: PCCSegment2D,
    center: np.ndarray,
    collision_radius: float,
    samples: int = 48,
) -> float | None:
    fractions = np.linspace(0.0, 1.0, samples)
    distances = np.array([np.linalg.norm(segment.point_at(fraction) - center) for fraction in fractions])
    inside = distances < collision_radius - 1e-4
    if not np.any(inside):
        return None

    first_inside = int(np.argmax(inside))
    if first_inside == 0:
        low, high = 0.0, fractions[1]
    else:
        low, high = fractions[first_inside - 1], fractions[first_inside]

    for _ in range(24):
        mid = 0.5 * (low + high)
        distance = np.linalg.norm(segment.point_at(mid) - center)
        if distance < collision_radius:
            high = mid
        else:
            low = mid
    return float(high)


def _minimum_distance_fraction(
    segment: PCCSegment2D,
    center: np.ndarray,
    samples: int = 33,
    refinements: int = 3,
) -> tuple[float, float]:
    low, high = 0.0, 1.0
    best_fraction = 0.0
    best_distance = float("inf")

    for _ in range(refinements):
        fractions = np.linspace(low, high, samples)
        distances = np.array([np.linalg.norm(segment.point_at(fraction) - center) for fraction in fractions])
        index = int(np.argmin(distances))
        best_fraction = float(fractions[index])
        best_distance = float(distances[index])
        left = max(0, index - 1)
        right = min(len(fractions) - 1, index + 1)
        low, high = float(fractions[left]), float(fractions[right])

    return best_fraction, best_distance


def detect_collision_event(
    segment: PCCSegment2D,
    obstacle: CircleObstacle,
    obstacle_index: int,
    segment_index: int,
    arm_radius: float,
) -> CollisionEvent | None:
    collision_radius = obstacle.radius + float(arm_radius)
    contact_fraction = _first_contact_fraction(segment, obstacle.center, collision_radius)
    if contact_fraction is None:
        return None

    minimum_fraction, minimum_distance = _minimum_distance_fraction(segment, obstacle.center)
    penetration = max(0.0, collision_radius - minimum_distance)
    nearest_point = segment.point_at(contact_fraction)
    tangent = normalize(segment.tangent_at(contact_fraction))
    radial = nearest_point - obstacle.center
    normal = normalize(radial, fallback=left_normal(tangent))
    contact_point = obstacle.center + normal * collision_radius

    return CollisionEvent(
        obstacle_index=obstacle_index,
        segment_index=segment_index,
        fraction=contact_fraction,
        nearest_point=nearest_point,
        contact_point=contact_point,
        tangent=tangent,
        normal=normal,
        penetration=penetration,
    )


def find_first_collision(
    segments: list[PCCSegment2D],
    obstacles: list[CircleObstacle],
    arm_radius: float,
) -> CollisionEvent | None:
    best_event: CollisionEvent | None = None
    for segment_index, segment in enumerate(segments):
        for obstacle_index, obstacle in enumerate(obstacles):
            event = detect_collision_event(
                segment=segment,
                obstacle=obstacle,
                obstacle_index=obstacle_index,
                segment_index=segment_index,
                arm_radius=arm_radius,
            )
            if event is None:
                continue
            if best_event is None or event.fraction < best_event.fraction:
                best_event = event
        if best_event is not None:
            return best_event
    return None


def apply_collision_response(
    segments: list[PCCSegment2D],
    event: CollisionEvent,
    obstacle: CircleObstacle,
    arm_radius: float,
    min_segment_length: float,
) -> list[PCCSegment2D]:
    original = segments[event.segment_index]
    front_length = original.length * np.clip(event.fraction, 0.0, 1.0)
    remaining_length = max(0.0, original.length - front_length)
    contact_radius = obstacle.radius + arm_radius

    desired_contact_length = max(
        min_segment_length,
        0.75 * obstacle.radius + 1.5 * event.penetration,
    )
    contact_length = min(remaining_length * 0.65, desired_contact_length)
    back_length = max(0.0, remaining_length - contact_length)

    if remaining_length <= min_segment_length:
        return segments

    tangent_candidate_a = left_normal(event.normal)
    tangent_candidate_b = -tangent_candidate_a
    start_tangent = (
        tangent_candidate_a
        if np.dot(tangent_candidate_a, event.tangent) >= np.dot(tangent_candidate_b, event.tangent)
        else tangent_candidate_b
    )
    center_offset = obstacle.center - event.contact_point
    curvature_sign = 1.0 if np.dot(left_normal(start_tangent), center_offset) >= 0.0 else -1.0
    contact_curvature = curvature_sign / max(contact_radius, EPSILON)

    replacement: list[PCCSegment2D] = []
    if front_length >= min_segment_length:
        replacement.append(PCCSegment2D(length=front_length, kappa=original.kappa))
    if contact_length >= min_segment_length:
        replacement.append(
            PCCSegment2D(
                length=contact_length,
                kappa=contact_curvature,
                start_direction=start_tangent,
                lock_start_direction=True,
            )
        )
    if back_length >= min_segment_length:
        replacement.append(PCCSegment2D(length=back_length, kappa=original.kappa))
    if not replacement:
        replacement.append(original.copy())

    return segments[: event.segment_index] + replacement + segments[event.segment_index + 1 :]
