from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TDCRRobotParams:
    segment_lengths: tuple[float, ...] = (0.08, 0.08, 0.08)
    tendons_per_segment: int = 3
    tendon_routing_radius: float = 0.006
    backbone_radius: float = 0.0007
    arm_radius: float = 0.004
    youngs_modulus: float = 60e9
    curvature_gain: float = 18.0
    torsion_gain: float = 0.15
    curvature_limit: float = 35.0
    samples_per_segment: int = 16

    @property
    def num_segments(self) -> int:
        return len(self.segment_lengths)

    @property
    def total_length(self) -> float:
        return float(sum(self.segment_lengths))

    @property
    def tau_dim(self) -> int:
        return self.num_segments * int(self.tendons_per_segment)

    def tendon_angles(self) -> np.ndarray:
        count = int(self.tendons_per_segment)
        if count != 3:
            return np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return np.deg2rad(np.array([90.0, -30.0, 210.0], dtype=float))

    def tendon_routing_vectors(self) -> np.ndarray:
        angles = self.tendon_angles()
        radius = float(self.tendon_routing_radius)
        return np.column_stack(
            (
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros_like(angles),
            )
        ).astype(float)

    def normalized_params(self) -> np.ndarray:
        return np.array(
            [
                self.total_length,
                float(np.mean(self.segment_lengths)),
                float(self.tendon_routing_radius),
                float(self.backbone_radius),
                float(self.arm_radius),
                float(self.youngs_modulus) / 1e9,
                float(self.curvature_gain),
                float(self.torsion_gain),
                float(self.curvature_limit),
                float(self.samples_per_segment),
            ],
            dtype=np.float32,
        )

    def validate_tau(self, tau: np.ndarray) -> np.ndarray:
        values = np.asarray(tau, dtype=float)
        if values.shape == (self.tau_dim,):
            values = values.reshape(self.num_segments, self.tendons_per_segment)
        expected = (self.num_segments, self.tendons_per_segment)
        if values.shape != expected:
            raise ValueError(f"Expected tau with shape {expected}, got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError("tau must contain only finite values.")
        return np.maximum(values, 0.0)

