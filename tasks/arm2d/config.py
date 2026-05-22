from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize_signed(value: float, scale: float) -> float:
    safe_scale = max(float(scale), 1e-8)
    return float(value) / safe_scale


def _normalize_unit_interval(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return 2.0 * ((float(value) - low) / (high - low)) - 1.0


def _safe_scaled_array(values: np.ndarray, scale: float, fill_value: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    safe_values = np.where(np.isfinite(values), values, fill_value)
    return (safe_values / max(float(scale), 1e-8)).astype(np.float32)


@dataclass(frozen=True)
class TendonControl2DConfig:
    base_delta_scale: float = 1.0
    tendon_delta_scale: float = 0.1
    action_low: float = -1.0
    action_high: float = 1.0
    allow_negative_base_delta: bool = False

    def action_dim(self, num_segments: int) -> int:
        return 1 + 2 * int(num_segments)

    def action_labels(self, num_segments: int) -> list[str]:
        labels = ["base_motor"]
        for index in range(num_segments):
            labels.extend([f"seg{index}_tendon_left_delta", f"seg{index}_tendon_right_delta"])
        return labels

    def clip_action(self, action: np.ndarray, num_segments: int) -> np.ndarray:
        clipped = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_size = self.action_dim(num_segments)
        if clipped.size != expected_size:
            raise ValueError(f"Expected action with {expected_size} values, got {clipped.size}.")
        return np.clip(clipped, self.action_low, self.action_high)

    def decode_action(self, action: np.ndarray, num_segments: int) -> tuple[np.ndarray, float, np.ndarray]:
        clipped = self.clip_action(action, num_segments)
        if self.allow_negative_base_delta:
            base_delta = float(clipped[0]) * self.base_delta_scale
        else:
            base_delta = max(float(clipped[0]), 0.0) * self.base_delta_scale
        tendon_length_deltas = clipped[1:].reshape(num_segments, 2) * self.tendon_delta_scale
        return clipped, base_delta, tendon_length_deltas.astype(np.float64, copy=False)


@dataclass(frozen=True)
class Reward2DConfig:
    progress_reward_weight: float = 2.0
    distance_reward_weight: float = 0.4
    angle_progress_reward_weight: float = 0.6
    angle_error_reward_weight: float = 0.2
    action_penalty_weight: float = 0.01
    time_penalty_weight: float = 0.01
    contact_progress_bonus_weight: float = 0.5
    idle_contact_penalty_weight: float = 0.05
    overlap_penalty_weight: float = 0.5
    success_bonus: float = 10.0
    progress_deadband: float = 1e-4

    def compute_process_terms(
        self,
        *,
        contact_summary,
        max_reach: float,
        dt_scale: float = 1.0,
    ) -> dict[str, float]:
        reach = max(float(max_reach), 1e-8)
        scale = max(float(dt_scale), 0.0)
        overlap_penalty = -self.overlap_penalty_weight * max(-float(contact_summary.min_clearance) / reach, 0.0) * scale
        time_penalty = -self.time_penalty_weight * scale
        return {
            "time_penalty": time_penalty,
            "overlap_penalty": overlap_penalty,
            "process_total": time_penalty + overlap_penalty,
        }

    def compute_terminal_terms(
        self,
        *,
        action: np.ndarray,
        distance: float,
        progress: float,
        angle_error: float,
        angle_progress: float,
        success: bool,
        contact_summary,
        max_reach: float,
    ) -> dict[str, float]:
        reach = max(float(max_reach), 1e-8)
        normalized_progress = float(progress) / reach
        normalized_distance = float(distance) / reach
        normalized_angle_progress = float(angle_progress) / np.pi
        normalized_angle_error = abs(float(angle_error)) / np.pi
        action_scale = max(np.sqrt(float(action.size)), 1.0)
        action_penalty = -self.action_penalty_weight * float(np.linalg.norm(action) / action_scale)
        progress_term = self.progress_reward_weight * normalized_progress
        distance_term = -self.distance_reward_weight * normalized_distance
        angle_progress_term = self.angle_progress_reward_weight * normalized_angle_progress
        angle_error_term = -self.angle_error_reward_weight * normalized_angle_error
        contact_activity = float(
            getattr(contact_summary, "contact_occurrence_rate", float(contact_summary.active))
        )
        positive_progress = max(normalized_progress, 0.0)
        contact_progress_bonus = self.contact_progress_bonus_weight * contact_activity * positive_progress
        idle_contact_penalty = (
            -self.idle_contact_penalty_weight
            if bool(contact_summary.active) and normalized_progress <= self.progress_deadband
            else 0.0
        )
        success_bonus = self.success_bonus if success else 0.0
        total = (
            progress_term
            + distance_term
            + angle_progress_term
            + angle_error_term
            + action_penalty
            + contact_progress_bonus
            + idle_contact_penalty
            + success_bonus
        )
        return {
            "progress": progress_term,
            "distance": distance_term,
            "angle_progress": angle_progress_term,
            "angle_error": angle_error_term,
            "action_penalty": action_penalty,
            "contact_progress_bonus": contact_progress_bonus,
            "idle_contact_penalty": idle_contact_penalty,
            "success_bonus": success_bonus,
            "normalized_distance": normalized_distance,
            "normalized_progress": normalized_progress,
            "normalized_angle_error": normalized_angle_error,
            "normalized_angle_progress": normalized_angle_progress,
            "terminal_total": total,
        }

    def compute_terms(
        self,
        *,
        action: np.ndarray,
        distance: float,
        progress: float,
        angle_error: float,
        angle_progress: float,
        success: bool,
        contact_summary,
        max_reach: float,
    ) -> dict[str, float]:
        process_terms = self.compute_process_terms(
            contact_summary=contact_summary,
            max_reach=max_reach,
            dt_scale=1.0,
        )
        terminal_terms = self.compute_terminal_terms(
            action=action,
            distance=distance,
            progress=progress,
            angle_error=angle_error,
            angle_progress=angle_progress,
            success=success,
            contact_summary=contact_summary,
            max_reach=max_reach,
        )
        return {
            **terminal_terms,
            "time_penalty": process_terms["time_penalty"],
            "overlap_penalty": process_terms["overlap_penalty"],
            "process_total": process_terms["process_total"],
            "terminal_total": terminal_terms["terminal_total"],
            "total": float(process_terms["process_total"] + terminal_terms["terminal_total"]),
        }


@dataclass(frozen=True)
class Observation2DConfig:
    mode: str = "hybrid"
    include_target: bool = True
    include_target_orientation: bool = True
    include_base_x: bool = True
    include_kappas: bool = True
    include_obstacles: bool = True
    include_contact_features: bool = True
    include_backbone_points: bool = True
    backbone_points: int = 4
    include_segment_contact_features: bool = True
    include_collision_count: bool = True
    normalize: bool = True
    low: float = -2.5
    high: float = 2.5

    def _base_dim(self) -> int:
        target_position_dim = 2 if self.include_target else 0
        target_orientation_dim = 2 if (self.include_target and self.include_target_orientation and self.mode != "legacy_v1") else 0
        if self.mode == "legacy_v1":
            return 3 + target_position_dim
        if self.mode == "absolute":
            return 4 + target_position_dim + target_orientation_dim
        if self.mode == "relative":
            return 2 + target_position_dim + target_orientation_dim
        if self.mode == "hybrid":
            return 4 + target_position_dim + target_orientation_dim
        raise ValueError(f"Unsupported observation mode {self.mode!r}.")

    def observation_dim(self, *, num_kappas: int, max_obstacles: int) -> int:
        dim = self._base_dim()
        if self.include_base_x:
            dim += 1
        if self.include_kappas:
            dim += int(num_kappas)
        if self.include_obstacles:
            obstacle_width = 3 if self.mode == "legacy_v1" else 4
            dim += obstacle_width * int(max_obstacles)
        if self.include_contact_features:
            dim += 7
        if self.include_backbone_points:
            dim += 2 * int(self.backbone_points)
        if self.include_segment_contact_features:
            dim += 2 * int(num_kappas)
        if self.include_collision_count:
            dim += 1
        return dim

    def build_observation(
        self,
        *,
        tip_state: np.ndarray,
        target: np.ndarray,
        target_angle: float,
        base_x: float,
        kappas: np.ndarray,
        obstacles,
        contact_summary,
        backbone_points: np.ndarray,
        collision_count: int,
        max_obstacles: int,
        max_reach: float,
        base_x_limits: tuple[float, float],
        curvature_limit: float,
    ) -> np.ndarray:
        tip_state = np.asarray(tip_state, dtype=np.float32)
        tip_position = tip_state[:2].astype(np.float32)
        tip_angle = float(tip_state[2])
        heading = np.array([np.cos(tip_angle), np.sin(tip_angle)], dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        target_heading = np.array([np.cos(float(target_angle)), np.sin(float(target_angle))], dtype=np.float32)
        target_delta = target - tip_position
        position_scale = max(float(max_reach), 1e-8)

        if self.normalize:
            tip_position_feat = (tip_position / position_scale).astype(np.float32)
            target_position_feat = (target / position_scale).astype(np.float32)
            target_delta_feat = (target_delta / position_scale).astype(np.float32)
        else:
            tip_position_feat = tip_position
            target_position_feat = target
            target_delta_feat = target_delta

        if self.mode == "legacy_v1":
            if self.normalize:
                tip_angle_feature = _normalize_signed(tip_angle, np.pi)
            else:
                tip_angle_feature = tip_angle
            parts = [np.array([tip_position_feat[0], tip_position_feat[1], tip_angle_feature], dtype=np.float32)]
            if self.include_target:
                parts.append(target_position_feat)
        elif self.mode == "absolute":
            parts = [tip_position_feat, heading]
            if self.include_target:
                parts.append(target_position_feat)
                if self.include_target_orientation:
                    parts.append(target_heading)
        elif self.mode == "relative":
            parts = [heading]
            if self.include_target:
                parts.insert(0, target_delta_feat)
                if self.include_target_orientation:
                    parts.append(target_heading)
        elif self.mode == "hybrid":
            parts = [tip_position_feat, heading]
            if self.include_target:
                parts.insert(1, target_delta_feat)
                if self.include_target_orientation:
                    parts.append(target_heading)
        else:
            raise ValueError(f"Unsupported observation mode {self.mode!r}.")

        if self.include_base_x:
            base_feature = (
                _normalize_unit_interval(base_x, base_x_limits[0], base_x_limits[1])
                if self.normalize
                else float(base_x)
            )
            parts.append(np.array([base_feature], dtype=np.float32))

        if self.include_kappas:
            kappas = np.asarray(kappas, dtype=np.float32)
            if self.normalize:
                safe_limit = max(float(curvature_limit), 1e-8)
                kappas = kappas / safe_limit
            parts.append(kappas.astype(np.float32))

        if self.include_obstacles:
            tip = tip_position.astype(float)
            if self.mode == "legacy_v1":
                sorted_obstacles = list(obstacles)
            else:
                sorted_obstacles = sorted(
                    list(obstacles),
                    key=lambda obstacle: np.linalg.norm(obstacle.center - tip) - obstacle.radius,
                )
            obstacle_values: list[float] = []
            for obstacle in sorted_obstacles[:max_obstacles]:
                if self.mode == "legacy_v1":
                    center = obstacle.center.astype(float)
                    if self.normalize:
                        center = center / position_scale
                        radius = obstacle.radius / position_scale
                    else:
                        radius = obstacle.radius
                    obstacle_values.extend((float(center[0]), float(center[1]), float(radius)))
                else:
                    relative = obstacle.center - tip
                    clearance = np.linalg.norm(relative) - obstacle.radius
                    if self.normalize:
                        relative = relative / position_scale
                        radius = obstacle.radius / position_scale
                        clearance = clearance / position_scale
                    else:
                        radius = obstacle.radius
                    obstacle_values.extend((float(relative[0]), float(relative[1]), float(radius), float(clearance)))
            for _ in range(max_obstacles - len(sorted_obstacles[:max_obstacles])):
                obstacle_values.extend((0.0, 0.0, 0.0) if self.mode == "legacy_v1" else (0.0, 0.0, 0.0, 0.0))
            parts.append(np.asarray(obstacle_values, dtype=np.float32))

        if self.include_contact_features:
            if contact_summary.nearest_obstacle_index >= 0:
                nearest_relative = contact_summary.nearest_obstacle_center - tip_position.astype(float)
                radius = contact_summary.nearest_obstacle_radius
            else:
                nearest_relative = np.zeros(2, dtype=float)
                radius = 0.0
            clearance = contact_summary.min_clearance
            if not np.isfinite(clearance):
                clearance = position_scale
            penetration = max(float(contact_summary.primary_penetration), 0.0)
            if self.normalize:
                nearest_relative = nearest_relative / position_scale
                clearance = clearance / position_scale
                penetration = penetration / position_scale
                radius = radius / position_scale
            parts.append(
                np.array(
                    [
                        float(getattr(contact_summary, "contact_occurrence_rate", float(contact_summary.active))),
                        float(clearance),
                        float(penetration),
                        float(contact_summary.nearest_fraction),
                        float(radius),
                        float(nearest_relative[0]),
                        float(nearest_relative[1]),
                    ],
                    dtype=np.float32,
                )
            )

        if self.include_backbone_points:
            backbone_points = np.asarray(backbone_points, dtype=np.float32)
            if backbone_points.size == 0:
                backbone_values = np.zeros(2 * int(self.backbone_points), dtype=np.float32)
            else:
                relative_backbone = backbone_points - tip_position[None, :]
                if self.normalize:
                    relative_backbone = relative_backbone / position_scale
                backbone_values = relative_backbone.reshape(-1).astype(np.float32)
            parts.append(backbone_values)

        if self.include_segment_contact_features:
            control_segments = int(np.asarray(kappas, dtype=np.float32).size)
            raw_clearances = np.asarray(contact_summary.segment_clearances, dtype=np.float32)
            raw_masks = np.asarray(contact_summary.segment_contact_mask, dtype=np.float32)
            segment_clearances = np.full(control_segments, np.inf, dtype=np.float32)
            segment_masks = np.zeros(control_segments, dtype=np.float32)
            usable = min(control_segments, raw_clearances.size)
            if usable:
                segment_clearances[:usable] = raw_clearances[:usable]
                segment_masks[:usable] = raw_masks[:usable]
            if self.normalize:
                segment_clearances = _safe_scaled_array(segment_clearances, position_scale, position_scale)
            else:
                segment_clearances = np.where(np.isfinite(segment_clearances), segment_clearances, position_scale)
            parts.append(segment_clearances.astype(np.float32))
            parts.append(segment_masks.astype(np.float32))

        if self.include_collision_count:
            count_value = float(collision_count)
            if self.normalize:
                count_value = min(count_value, float(max_obstacles)) / max(float(max_obstacles), 1.0)
            parts.append(np.array([count_value], dtype=np.float32))

        observation = np.concatenate(parts, dtype=np.float32)
        return np.clip(observation, self.low, self.high)


@dataclass(frozen=True)
class Render2DConfig:
    xlim: tuple[float, float] = (-2.0, 42.0)
    ylim: tuple[float, float] = (-20.0, 20.0)
    figsize: tuple[float, float] = (8.0, 5.0)
    title: str = "Soft Arm Obstacle 2D"
    show_grid: bool = True


@dataclass(frozen=True)
class SoftArmObstacle2DEnvConfig:
    control: TendonControl2DConfig = field(default_factory=TendonControl2DConfig)
    reward: Reward2DConfig = field(default_factory=Reward2DConfig)
    observation: Observation2DConfig = field(default_factory=Observation2DConfig)
    render: Render2DConfig = field(default_factory=Render2DConfig)
