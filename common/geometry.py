from __future__ import annotations

import numpy as np

EPSILON = 1e-8


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= EPSILON:
        if fallback is None:
            fallback = np.array([1.0, 0.0], dtype=float)
        return np.asarray(fallback, dtype=float).copy()
    return array / norm


def left_normal(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    return np.array([-direction[1], direction[0]], dtype=float)


def rotation_from_direction(direction: np.ndarray) -> np.ndarray:
    tangent = normalize(direction)
    normal = left_normal(tangent)
    return np.column_stack((tangent, normal))


def angle_of(direction: np.ndarray) -> float:
    tangent = normalize(direction)
    return float(np.arctan2(tangent[1], tangent[0]))
