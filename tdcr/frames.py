from __future__ import annotations

import numpy as np

EPSILON = 1e-9


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= EPSILON:
        if fallback is None:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        return np.asarray(fallback, dtype=float).copy()
    return value / norm


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    omega = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta <= EPSILON:
        return np.eye(3, dtype=float) + hat(omega)
    axis_hat = hat(omega / theta)
    return np.eye(3, dtype=float) + np.sin(theta) * axis_hat + (1.0 - np.cos(theta)) * (axis_hat @ axis_hat)


def rotvec_from_matrix(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace_value = float(np.trace(rotation))
    cos_theta = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta <= 1e-8:
        return np.zeros(3, dtype=float)
    skew = (rotation - rotation.T) / (2.0 * np.sin(theta))
    return theta * np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


def transform_from_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return transform


def project_xz(points: np.ndarray) -> np.ndarray:
    samples = np.asarray(points, dtype=float)
    return np.column_stack((samples[..., 2].reshape(-1), samples[..., 0].reshape(-1)))

