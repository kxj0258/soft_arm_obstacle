from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdcr.actuation import TensionControlConfig


def _normalize_vector(values: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) / max(float(scale), 1e-8)).astype(np.float32)


def _safe_scaled_array(values: np.ndarray, scale: float, fill_value: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    safe_values = np.where(np.isfinite(values), values, fill_value)
    return (safe_values / max(float(scale), 1e-8)).astype(np.float32)


@dataclass(frozen=True)
class Reward3DConfig:
    progress_reward_weight: float = 2.0
    distance_reward_weight: float = 0.55
    orientation_progress_reward_weight: float = 0.5
    orientation_error_reward_weight: float = 0.18
    action_penalty_weight: float = 0.012
    tension_penalty_weight: float = 0.004
    time_penalty_weight: float = 0.01
    contact_progress_bonus_weight: float = 0.35
    idle_contact_penalty_weight: float = 0.05
    penetration_penalty_weight: float = 1.2
    success_bonus: float = 10.0
    progress_deadband: float = 1e-5

    def compute_process_terms(
        self,
        *,
        contact_summary,
        max_reach: float,
        dt_scale: float = 1.0,
    ) -> dict[str, float]:
        reach = max(float(max_reach), 1e-8)
        scale = max(float(dt_scale), 0.0)
        penetration_penalty = -self.penetration_penalty_weight * max(
            float(contact_summary.primary_penetration) / reach,
            0.0,
        ) * scale
        time_penalty = -self.time_penalty_weight * scale
        return {
            "time_penalty": time_penalty,
            "penetration_penalty": penetration_penalty,
            "process_total": time_penalty + penetration_penalty,
        }

    def compute_terminal_terms(
        self,
        *,
        action: np.ndarray,
        tau: np.ndarray,
        tau_max: float,
        distance: float,
        progress: float,
        orientation_error: float,
        orientation_progress: float,
        success: bool,
        contact_summary,
        max_reach: float,
    ) -> dict[str, float]:
        reach = max(float(max_reach), 1e-8)
        normalized_progress = float(progress) / reach
        normalized_distance = float(distance) / reach
        normalized_orientation_error = float(abs(orientation_error)) / np.pi
        normalized_orientation_progress = float(orientation_progress) / np.pi
        action_scale = max(np.sqrt(float(action.size)), 1.0)
        tau_scale = max(float(tau_max), 1e-8) * max(np.sqrt(float(np.asarray(tau).size)), 1.0)

        action_penalty = -self.action_penalty_weight * float(np.linalg.norm(action) / action_scale)
        tension_penalty = -self.tension_penalty_weight * float(np.linalg.norm(tau) / tau_scale)
        progress_term = self.progress_reward_weight * normalized_progress
        distance_term = -self.distance_reward_weight * normalized_distance
        orientation_progress_term = self.orientation_progress_reward_weight * normalized_orientation_progress
        orientation_error_term = -self.orientation_error_reward_weight * normalized_orientation_error
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
            + orientation_progress_term
            + orientation_error_term
            + action_penalty
            + tension_penalty
            + contact_progress_bonus
            + idle_contact_penalty
            + success_bonus
        )
        return {
            "progress": progress_term,
            "distance": distance_term,
            "orientation_progress": orientation_progress_term,
            "orientation_error": orientation_error_term,
            "action_penalty": action_penalty,
            "tension_penalty": tension_penalty,
            "contact_progress_bonus": contact_progress_bonus,
            "idle_contact_penalty": idle_contact_penalty,
            "success_bonus": success_bonus,
            "normalized_distance": normalized_distance,
            "normalized_progress": normalized_progress,
            "normalized_orientation_error": normalized_orientation_error,
            "normalized_orientation_progress": normalized_orientation_progress,
            "terminal_total": total,
        }

    def compute_terms(
        self,
        *,
        action: np.ndarray,
        tau: np.ndarray,
        tau_max: float,
        distance: float,
        progress: float,
        orientation_error: float,
        orientation_progress: float,
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
            tau=tau,
            tau_max=tau_max,
            distance=distance,
            progress=progress,
            orientation_error=orientation_error,
            orientation_progress=orientation_progress,
            success=success,
            contact_summary=contact_summary,
            max_reach=max_reach,
        )
        return {
            **terminal_terms,
            "time_penalty": process_terms["time_penalty"],
            "penetration_penalty": process_terms["penetration_penalty"],
            "process_total": process_terms["process_total"],
            "terminal_total": terminal_terms["terminal_total"],
            "total": float(process_terms["process_total"] + terminal_terms["terminal_total"]),
        }


@dataclass(frozen=True)
class Observation3DConfig:
    include_target: bool = True
    include_orientation: bool = True
    include_tau: bool = True
    include_strain: bool = True
    include_obstacles: bool = True
    include_contact_features: bool = True
    include_backbone_points: bool = True
    backbone_points: int = 6
    include_segment_contact_features: bool = True
    include_collision_count: bool = True
    normalize: bool = True
    low: float = -5.0
    high: float = 5.0

    def observation_dim(self, *, tau_dim: int, num_segments: int, max_obstacles: int) -> int:
        dim = 3  # tip position
        if self.include_target:
            dim += 6  # target delta + target position
        if self.include_orientation:
            dim += 6  # tip rotvec + target rotvec
        if self.include_tau:
            dim += int(tau_dim)
        if self.include_strain:
            dim += int(num_segments) * 3
        if self.include_obstacles:
            dim += int(max_obstacles) * 8  # relative center, radii, clearance, active flag
        if self.include_contact_features:
            dim += 17
        if self.include_backbone_points:
            dim += int(self.backbone_points) * 3
        if self.include_segment_contact_features:
            dim += int(num_segments) * 2
        if self.include_collision_count:
            dim += 1
        return dim

    def build_observation(
        self,
        *,
        tip_position: np.ndarray,
        tip_rotvec: np.ndarray,
        target: np.ndarray,
        target_rotvec: np.ndarray,
        tau: np.ndarray,
        tau_max: float,
        strain_u_by_segment: np.ndarray,
        obstacles,
        contact_summary,
        backbone_points: np.ndarray,
        collision_count: int,
        max_obstacles: int,
        max_reach: float,
        curvature_limit: float,
    ) -> np.ndarray:
        tip_position = np.asarray(tip_position, dtype=np.float32).reshape(3)
        tip_rotvec = np.asarray(tip_rotvec, dtype=np.float32).reshape(3)
        target = np.asarray(target, dtype=np.float32).reshape(3)
        target_rotvec = np.asarray(target_rotvec, dtype=np.float32).reshape(3)
        target_delta = target - tip_position
        position_scale = max(float(max_reach), 1e-8)

        parts: list[np.ndarray] = []
        if self.normalize:
            parts.append(_normalize_vector(tip_position, position_scale))
        else:
            parts.append(tip_position.astype(np.float32))

        if self.include_target:
            if self.normalize:
                parts.append(_normalize_vector(target_delta, position_scale))
                parts.append(_normalize_vector(target, position_scale))
            else:
                parts.append(target_delta.astype(np.float32))
                parts.append(target.astype(np.float32))

        if self.include_orientation:
            if self.normalize:
                parts.append(_normalize_vector(tip_rotvec, np.pi))
                parts.append(_normalize_vector(target_rotvec, np.pi))
            else:
                parts.append(tip_rotvec.astype(np.float32))
                parts.append(target_rotvec.astype(np.float32))

        if self.include_tau:
            tau_values = np.asarray(tau, dtype=np.float32).reshape(-1)
            if self.normalize:
                tau_values = tau_values / max(float(tau_max), 1e-8)
            parts.append(tau_values.astype(np.float32))

        if self.include_strain:
            strains = np.asarray(strain_u_by_segment, dtype=np.float32).reshape(-1)
            if self.normalize:
                strains = strains / max(float(curvature_limit), 1e-8)
            parts.append(strains.astype(np.float32))

        if self.include_obstacles:
            sorted_obstacles = sorted(
                list(obstacles),
                key=lambda obstacle: np.min(obstacle.clearance(tip_position[None, :], inflation=0.0)),
            )
            obstacle_values: list[float] = []
            for obstacle in sorted_obstacles[:max_obstacles]:
                relative = obstacle.center.astype(float) - tip_position.astype(float)
                clearance = float(obstacle.clearance(tip_position[None, :], inflation=0.0)[0])
                radii = obstacle.radii.astype(float)
                if self.normalize:
                    relative = relative / position_scale
                    radii = radii / position_scale
                    clearance = clearance / position_scale
                obstacle_values.extend(
                    (
                        float(relative[0]),
                        float(relative[1]),
                        float(relative[2]),
                        float(radii[0]),
                        float(radii[1]),
                        float(radii[2]),
                        float(clearance),
                        1.0,
                    )
                )
            for _ in range(max_obstacles - len(sorted_obstacles[:max_obstacles])):
                obstacle_values.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            parts.append(np.asarray(obstacle_values, dtype=np.float32))

        if self.include_contact_features:
            if contact_summary.nearest_obstacle_index >= 0:
                nearest_relative = contact_summary.nearest_obstacle_center - tip_position.astype(float)
                nearest_point_relative = contact_summary.nearest_point - tip_position.astype(float)
                radii = contact_summary.nearest_obstacle_radii.astype(float)
            else:
                nearest_relative = np.zeros(3, dtype=float)
                nearest_point_relative = np.zeros(3, dtype=float)
                radii = np.zeros(3, dtype=float)
            clearance = float(contact_summary.min_clearance)
            if not np.isfinite(clearance):
                clearance = position_scale
            penetration = max(float(contact_summary.primary_penetration), 0.0)
            force = np.asarray(contact_summary.contact_force, dtype=float).reshape(3)
            if self.normalize:
                nearest_relative = nearest_relative / position_scale
                nearest_point_relative = nearest_point_relative / position_scale
                radii = radii / position_scale
                clearance = clearance / position_scale
                penetration = penetration / position_scale
                force = force / max(float(tau_max), 1e-8)
            parts.append(
                np.asarray(
                    [
                        float(contact_summary.active),
                        float(clearance),
                        float(penetration),
                        float(contact_summary.nearest_s) / position_scale,
                        *nearest_relative.tolist(),
                        *nearest_point_relative.tolist(),
                        *radii.tolist(),
                        *np.asarray(contact_summary.contact_normal, dtype=float).reshape(3).tolist(),
                        float(np.linalg.norm(force)),
                    ],
                    dtype=np.float32,
                )
            )

        if self.include_backbone_points:
            backbone_points = np.asarray(backbone_points, dtype=np.float32).reshape(-1, 3)
            if backbone_points.size == 0:
                backbone_values = np.zeros(3 * int(self.backbone_points), dtype=np.float32)
            else:
                relative_backbone = backbone_points - tip_position[None, :]
                if self.normalize:
                    relative_backbone = relative_backbone / position_scale
                backbone_values = relative_backbone.reshape(-1).astype(np.float32)
            parts.append(backbone_values)

        if self.include_segment_contact_features:
            raw_clearances = np.asarray(contact_summary.segment_clearances, dtype=np.float32)
            raw_masks = np.asarray(contact_summary.segment_contact_mask, dtype=np.float32)
            segment_clearances = np.full(int(num_segments := strain_u_by_segment.shape[0]), np.inf, dtype=np.float32)
            segment_masks = np.zeros(int(num_segments), dtype=np.float32)
            usable = min(int(num_segments), raw_clearances.size)
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
class Render3DConfig:
    xlim: tuple[float, float] = (-0.16, 0.16)
    ylim: tuple[float, float] = (-0.16, 0.16)
    zlim: tuple[float, float] = (0.0, 0.30)
    figsize: tuple[float, float] = (7.0, 6.0)
    title: str = "Soft Arm Obstacle 3D"
    show_grid: bool = True


@dataclass(frozen=True)
class SoftArmObstacle3DEnvConfig:
    control: TensionControlConfig = field(default_factory=TensionControlConfig)
    reward: Reward3DConfig = field(default_factory=Reward3DConfig)
    observation: Observation3DConfig = field(default_factory=Observation3DConfig)
    render: Render3DConfig = field(default_factory=Render3DConfig)
