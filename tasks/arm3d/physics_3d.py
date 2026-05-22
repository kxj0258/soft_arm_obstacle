from __future__ import annotations

from typing import Iterable

import numpy as np

from tdcr.shapes import ContactSummary3D, Obstacle3D


def _obstacle_from_mapping(payload: dict) -> Obstacle3D:
    center = payload.get("center")
    if center is None:
        center = (payload.get("x", 0.0), payload.get("y", 0.0), payload.get("z", 0.0))
    radii = payload.get("radii", payload.get("radius", payload.get("r", None)))
    if radii is None:
        raise ValueError("3D obstacle mappings must provide 'radii', 'radius', or 'r'.")
    return Obstacle3D(center=np.asarray(center, dtype=float), radii=radii, kind=str(payload.get("kind", "sphere")))


def ensure_obstacles(obstacles: Iterable[Obstacle3D | dict | tuple | list] | None) -> list[Obstacle3D]:
    """Normalize supported obstacle payloads into TDCR 3D obstacles.

    Supported tuple/list layouts are:
    - ``(x, y, z, radius)`` for spheres
    - ``(x, y, z, rx, ry, rz)`` for axis-aligned ellipsoids
    """

    normalized: list[Obstacle3D] = []
    for obstacle in obstacles or []:
        if isinstance(obstacle, Obstacle3D):
            normalized.append(Obstacle3D(center=obstacle.center, radii=obstacle.radii, kind=obstacle.kind))
            continue
        if isinstance(obstacle, dict):
            normalized.append(_obstacle_from_mapping(obstacle))
            continue
        values = np.asarray(obstacle, dtype=float).reshape(-1)
        if values.size == 4:
            normalized.append(Obstacle3D(center=values[:3], radii=float(values[3]), kind="sphere"))
        elif values.size == 6:
            normalized.append(Obstacle3D(center=values[:3], radii=values[3:6], kind="ellipsoid"))
        else:
            raise ValueError(
                "3D obstacles must be Obstacle3D, a mapping, (x, y, z, r), or (x, y, z, rx, ry, rz)."
            )
    return normalized


def make_empty_contact_summary(num_segments: int) -> ContactSummary3D:
    return ContactSummary3D(
        segment_clearances=np.full(int(num_segments), float("inf"), dtype=float),
        segment_contact_mask=np.zeros(int(num_segments), dtype=bool),
        contact_occurrence_rate=0.0,
        micro_step_count=1,
    )


__all__ = ["ContactSummary3D", "Obstacle3D", "ensure_obstacles", "make_empty_contact_summary"]
