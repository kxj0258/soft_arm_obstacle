from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def flatten_obstacles(obstacles) -> np.ndarray:
    values: list[float] = []
    for obstacle in obstacles:
        values.extend((float(obstacle.center[0]), float(obstacle.center[1]), float(obstacle.radius)))
    return np.asarray(values, dtype=np.float32)
