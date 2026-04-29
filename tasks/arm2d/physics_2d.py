from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from common.geometry import EPSILON, left_normal, normalize
from common.pcc import PCCSegment2D


@dataclass(frozen=True)
class ContactResponse2DConfig:
    backend: str = "heuristic"
    coarse_collision_samples: int = 36
    min_distance_samples: int = 17
    min_distance_refinements: int = 2
    broadphase_samples: int = 11
    broadphase_margin: float = 0.75
    contact_length_obstacle_factor: float = 0.75
    contact_length_penetration_factor: float = 1.5
    max_contact_fraction: float = 0.65
    contact_release_clearance: float = 0.4
    collision_epsilon: float = 1e-4
    optimization_subsegments_per_segment: int = 4
    optimization_tracking_weight: float = 0.25
    optimization_smoothness_weight: float = 0.05
    optimization_locality_margin_factor: float = 2.5
    optimization_max_iterations: int = 80
    optimization_feasibility_tolerance: float = 5e-5
    fallback_to_heuristic: bool = True


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float
    superellipse_power: float = 2.0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).copy()
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError("Obstacle radius must be positive.")
        self.superellipse_power = float(self.superellipse_power)
        if self.superellipse_power <= 0.0:
            raise ValueError("Obstacle superellipse_power must be positive.")

    def implicit_values(self, points: np.ndarray, *, inflation: float = 0.0) -> np.ndarray:
        samples = np.asarray(points, dtype=float)
        flat_samples = samples.reshape(-1, 2)
        scale = max(self.radius + float(inflation), EPSILON)
        normalized = np.abs(flat_samples - self.center[None, :]) / scale
        values = np.sum(normalized ** self.superellipse_power, axis=1) - 1.0
        if samples.ndim == 1:
            return np.asarray(values[0], dtype=float)
        return values.astype(float)

    def clearance_values(self, points: np.ndarray, *, inflation: float = 0.0) -> np.ndarray:
        samples = np.asarray(points, dtype=float)
        flat_samples = samples.reshape(-1, 2)
        clearance = np.linalg.norm(flat_samples - self.center[None, :], axis=1) - (self.radius + float(inflation))
        if samples.ndim == 1:
            return np.asarray(clearance[0], dtype=float)
        return clearance.astype(float)


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


@dataclass
class ClearanceRecord:
    obstacle_index: int
    segment_index: int
    fraction: float
    arm_point: np.ndarray
    obstacle_point: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: float
    signed_clearance: float


@dataclass
class ContactSummary:
    active: bool
    contact_state: str
    collision_count: int
    min_clearance: float
    nearest_obstacle_index: int
    nearest_segment_index: int
    nearest_fraction: float
    nearest_arm_point: np.ndarray
    nearest_obstacle_point: np.ndarray
    nearest_obstacle_center: np.ndarray
    nearest_obstacle_radius: float
    primary_penetration: float
    segment_clearances: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    segment_contact_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "contact_state": self.contact_state,
            "collision_count": int(self.collision_count),
            "min_clearance": float(self.min_clearance),
            "nearest_obstacle_index": int(self.nearest_obstacle_index),
            "nearest_segment_index": int(self.nearest_segment_index),
            "nearest_fraction": float(self.nearest_fraction),
            "nearest_arm_point": self.nearest_arm_point.copy(),
            "nearest_obstacle_point": self.nearest_obstacle_point.copy(),
            "nearest_obstacle_center": self.nearest_obstacle_center.copy(),
            "nearest_obstacle_radius": float(self.nearest_obstacle_radius),
            "primary_penetration": float(self.primary_penetration),
            "segment_clearances": self.segment_clearances.copy(),
            "segment_contact_mask": self.segment_contact_mask.copy(),
        }


def make_empty_contact_summary(num_segments: int = 0) -> ContactSummary:
    return ContactSummary(
        active=False,
        contact_state="free",
        collision_count=0,
        min_clearance=float("inf"),
        nearest_obstacle_index=-1,
        nearest_segment_index=-1,
        nearest_fraction=0.0,
        nearest_arm_point=np.zeros(2, dtype=float),
        nearest_obstacle_point=np.zeros(2, dtype=float),
        nearest_obstacle_center=np.zeros(2, dtype=float),
        nearest_obstacle_radius=0.0,
        primary_penetration=0.0,
        segment_clearances=np.full(num_segments, float("inf"), dtype=float),
        segment_contact_mask=np.zeros(num_segments, dtype=bool),
    )


def ensure_obstacles(obstacles: Iterable[CircleObstacle | tuple[float, float, float]]) -> list[CircleObstacle]:
    normalized: list[CircleObstacle] = []
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            normalized.append(
                CircleObstacle(
                    obstacle.center,
                    obstacle.radius,
                    superellipse_power=obstacle.superellipse_power,
                )
            )
        else:
            x, y, radius = obstacle
            normalized.append(CircleObstacle(center=np.array([x, y], dtype=float), radius=radius))
    return normalized


def _segment_points(
    segment: PCCSegment2D,
    sample_count: int,
    sample_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    key = (id(segment), int(sample_count))
    cached = sample_cache.get(key)
    if cached is None:
        cached = segment.sample_points(sample_count)
        sample_cache[key] = cached
    return cached


def _bbox_distance_to_point(bounds_min: np.ndarray, bounds_max: np.ndarray, point: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(bounds_min - point, point - bounds_max))
    return float(np.linalg.norm(delta))


def _segment_obstacle_broadphase(
    segment: PCCSegment2D,
    obstacle: CircleObstacle,
    interaction_radius: float,
    response_config: ContactResponse2DConfig,
    sample_cache: dict[tuple[int, int], np.ndarray],
) -> bool:
    points = _segment_points(segment, response_config.broadphase_samples, sample_cache)
    bounds_min = np.min(points, axis=0) - response_config.broadphase_margin
    bounds_max = np.max(points, axis=0) + response_config.broadphase_margin
    return _bbox_distance_to_point(bounds_min, bounds_max, obstacle.center) <= interaction_radius


def _first_contact_fraction(
    segment: PCCSegment2D,
    center: np.ndarray,
    collision_radius: float,
    response_config: ContactResponse2DConfig,
) -> float | None:
    fractions = np.linspace(0.0, 1.0, response_config.coarse_collision_samples)
    positions = segment.pose_at_fractions(fractions)[0]
    distances = np.linalg.norm(positions - center[None, :], axis=1)
    inside = distances < collision_radius - response_config.collision_epsilon
    if not np.any(inside):
        return None

    first_inside = int(np.argmax(inside))
    if first_inside == 0:
        low, high = 0.0, fractions[min(1, len(fractions) - 1)]
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
    response_config: ContactResponse2DConfig,
) -> tuple[float, float]:
    low, high = 0.0, 1.0
    best_fraction = 0.0
    best_distance = float("inf")

    for _ in range(response_config.min_distance_refinements):
        fractions = np.linspace(low, high, response_config.min_distance_samples)
        positions = segment.pose_at_fractions(fractions)[0]
        distances = np.linalg.norm(positions - center[None, :], axis=1)
        index = int(np.argmin(distances))
        best_fraction = float(fractions[index])
        best_distance = float(distances[index])
        left = max(0, index - 1)
        right = min(len(fractions) - 1, index + 1)
        low, high = float(fractions[left]), float(fractions[right])

    return best_fraction, best_distance


def measure_segment_clearance(
    segment: PCCSegment2D,
    obstacle: CircleObstacle,
    obstacle_index: int,
    segment_index: int,
    arm_radius: float,
    response_config: ContactResponse2DConfig | None = None,
) -> ClearanceRecord:
    response_config = response_config or ContactResponse2DConfig()
    fraction, centerline_distance = _minimum_distance_fraction(segment, obstacle.center, response_config)
    arm_point = segment.point_at(fraction)
    tangent = normalize(segment.tangent_at(fraction))
    radial = arm_point - obstacle.center
    radial_direction = normalize(radial, fallback=left_normal(tangent))
    obstacle_point = obstacle.center + radial_direction * obstacle.radius
    signed_clearance = centerline_distance - (obstacle.radius + float(arm_radius))
    return ClearanceRecord(
        obstacle_index=obstacle_index,
        segment_index=segment_index,
        fraction=fraction,
        arm_point=arm_point,
        obstacle_point=obstacle_point,
        obstacle_center=obstacle.center.copy(),
        obstacle_radius=obstacle.radius,
        signed_clearance=float(signed_clearance),
    )


def detect_collision_event(
    segment: PCCSegment2D,
    obstacle: CircleObstacle,
    obstacle_index: int,
    segment_index: int,
    arm_radius: float,
    response_config: ContactResponse2DConfig | None = None,
) -> CollisionEvent | None:
    response_config = response_config or ContactResponse2DConfig()
    collision_radius = obstacle.radius + float(arm_radius)
    contact_fraction = _first_contact_fraction(segment, obstacle.center, collision_radius, response_config)
    if contact_fraction is None:
        return None

    _, minimum_distance = _minimum_distance_fraction(segment, obstacle.center, response_config)
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
    response_config: ContactResponse2DConfig | None = None,
) -> CollisionEvent | None:
    response_config = response_config or ContactResponse2DConfig()
    sample_cache: dict[tuple[int, int], np.ndarray] = {}
    best_event: CollisionEvent | None = None
    for segment_index, segment in enumerate(segments):
        collision_radius = arm_radius
        for obstacle_index, obstacle in enumerate(obstacles):
            if not _segment_obstacle_broadphase(
                segment,
                obstacle,
                obstacle.radius + collision_radius,
                response_config,
                sample_cache,
            ):
                continue
            event = detect_collision_event(
                segment=segment,
                obstacle=obstacle,
                obstacle_index=obstacle_index,
                segment_index=segment_index,
                arm_radius=arm_radius,
                response_config=response_config,
            )
            if event is None:
                continue
            if best_event is None or event.fraction < best_event.fraction:
                best_event = event
        if best_event is not None:
            return best_event
    return None


def summarize_contacts(
    segments: list[PCCSegment2D],
    obstacles: list[CircleObstacle],
    arm_radius: float,
    collision_events: list[CollisionEvent] | None = None,
    response_config: ContactResponse2DConfig | None = None,
    previous_summary: ContactSummary | None = None,
    *,
    contact_activation_clearance: float | None = None,
    summary_segment_slices: tuple[slice, ...] | None = None,
) -> ContactSummary:
    response_config = response_config or ContactResponse2DConfig()
    collision_events = collision_events or []
    if not segments or not obstacles:
        if summary_segment_slices:
            return make_empty_contact_summary(len(summary_segment_slices))
        return make_empty_contact_summary(len(segments))

    best_record: ClearanceRecord | None = None
    detailed_segment_clearances = np.full(len(segments), float("inf"), dtype=float)
    for segment_index, segment in enumerate(segments):
        for obstacle_index, obstacle in enumerate(obstacles):
            record = measure_segment_clearance(
                segment=segment,
                obstacle=obstacle,
                obstacle_index=obstacle_index,
                segment_index=segment_index,
                arm_radius=arm_radius,
                response_config=response_config,
            )
            detailed_segment_clearances[segment_index] = min(
                detailed_segment_clearances[segment_index],
                record.signed_clearance,
            )
            if best_record is None or record.signed_clearance < best_record.signed_clearance:
                best_record = record

    assert best_record is not None
    colliding = bool(collision_events)
    close_contact = bool(
        contact_activation_clearance is not None
        and best_record.signed_clearance <= float(contact_activation_clearance)
    )
    latched = bool(
        previous_summary is not None
        and previous_summary.active
        and not colliding
        and best_record.obstacle_index == previous_summary.nearest_obstacle_index
        and best_record.signed_clearance <= response_config.contact_release_clearance
    )
    active = colliding or latched or close_contact
    contact_state = "collision" if colliding else ("latched" if latched else ("constrained" if close_contact else "free"))
    primary_penetration = collision_events[0].penetration if colliding else max(0.0, -best_record.signed_clearance)
    detailed_segment_contact_mask = detailed_segment_clearances <= response_config.contact_release_clearance

    if summary_segment_slices:
        segment_clearances = np.full(len(summary_segment_slices), float("inf"), dtype=float)
        segment_contact_mask = np.zeros(len(summary_segment_slices), dtype=bool)
        for summary_index, segment_slice in enumerate(summary_segment_slices):
            if segment_slice.start is None or segment_slice.stop is None or segment_slice.start >= segment_slice.stop:
                continue
            slice_clearances = detailed_segment_clearances[segment_slice]
            slice_mask = detailed_segment_contact_mask[segment_slice]
            if slice_clearances.size:
                segment_clearances[summary_index] = float(np.min(slice_clearances))
                segment_contact_mask[summary_index] = bool(np.any(slice_mask))
    else:
        segment_clearances = detailed_segment_clearances
        segment_contact_mask = detailed_segment_contact_mask

    return ContactSummary(
        active=active,
        contact_state=contact_state,
        collision_count=len(collision_events),
        min_clearance=float(best_record.signed_clearance),
        nearest_obstacle_index=best_record.obstacle_index,
        nearest_segment_index=best_record.segment_index,
        nearest_fraction=float(best_record.fraction),
        nearest_arm_point=best_record.arm_point.copy(),
        nearest_obstacle_point=best_record.obstacle_point.copy(),
        nearest_obstacle_center=best_record.obstacle_center.copy(),
        nearest_obstacle_radius=float(best_record.obstacle_radius),
        primary_penetration=float(primary_penetration),
        segment_clearances=segment_clearances,
        segment_contact_mask=segment_contact_mask,
    )


def apply_collision_response(
    segments: list[PCCSegment2D],
    event: CollisionEvent,
    obstacle: CircleObstacle,
    arm_radius: float,
    min_segment_length: float,
    response_config: ContactResponse2DConfig | None = None,
) -> list[PCCSegment2D]:
    response_config = response_config or ContactResponse2DConfig()
    original = segments[event.segment_index]
    front_length = original.length * np.clip(event.fraction, 0.0, 1.0)
    remaining_length = max(0.0, original.length - front_length)
    contact_radius = obstacle.radius + arm_radius

    desired_contact_length = max(
        min_segment_length,
        response_config.contact_length_obstacle_factor * obstacle.radius
        + response_config.contact_length_penetration_factor * event.penetration,
    )
    contact_length = min(remaining_length * response_config.max_contact_fraction, desired_contact_length)
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
