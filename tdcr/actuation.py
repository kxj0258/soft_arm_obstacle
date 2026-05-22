from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensionControlConfig:
    tau_min: float = 0.0
    tau_max: float = 3.5
    action_low: float = -1.0
    action_high: float = 1.0
    smoothing: float = 0.0

    def action_dim(self, num_segments: int, tendons_per_segment: int) -> int:
        return int(num_segments) * int(tendons_per_segment)

    def action_labels(self, num_segments: int, tendons_per_segment: int) -> list[str]:
        labels: list[str] = []
        for segment_index in range(int(num_segments)):
            for tendon_index in range(int(tendons_per_segment)):
                labels.append(f"seg{segment_index}_tendon{tendon_index}_tension")
        return labels

    def clip_action(self, action: np.ndarray, num_segments: int, tendons_per_segment: int) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_size = self.action_dim(num_segments, tendons_per_segment)
        if values.size != expected_size:
            raise ValueError(f"Expected action with {expected_size} values, got {values.size}.")
        return np.clip(values, self.action_low, self.action_high)

    def decode_action(
        self,
        action: np.ndarray,
        *,
        num_segments: int,
        tendons_per_segment: int,
        previous_tau: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = self.clip_action(action, num_segments, tendons_per_segment)
        span = float(self.tau_max) - float(self.tau_min)
        if span <= 0.0:
            raise ValueError("tau_max must be greater than tau_min.")
        normalized = (clipped - float(self.action_low)) / max(float(self.action_high - self.action_low), 1e-8)
        tau = float(self.tau_min) + normalized.reshape(num_segments, tendons_per_segment) * span
        if previous_tau is not None and self.smoothing > 0.0:
            prev = np.asarray(previous_tau, dtype=float).reshape(num_segments, tendons_per_segment)
            alpha = float(np.clip(self.smoothing, 0.0, 1.0))
            tau = alpha * prev + (1.0 - alpha) * tau
        return clipped.astype(np.float32, copy=False), tau.astype(float, copy=False)

