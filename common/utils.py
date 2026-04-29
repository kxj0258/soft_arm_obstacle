from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def flatten_obstacles(obstacles, expected_count: int | None = None) -> np.ndarray:
    obstacles = list(obstacles)
    if expected_count is not None and len(obstacles) > expected_count:
        raise ValueError(
            f"Expected at most {expected_count} obstacles for observation packing, got {len(obstacles)}."
        )

    values: list[float] = []
    for obstacle in obstacles:
        values.extend((float(obstacle.center[0]), float(obstacle.center[1]), float(obstacle.radius)))

    if expected_count is not None:
        for _ in range(expected_count - len(obstacles)):
            values.extend((0.0, 0.0, 0.0))
    return np.asarray(values, dtype=np.float32)
