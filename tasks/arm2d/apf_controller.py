from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import normalize

from .config import TendonControl2DConfig
from .physics_2d import CircleObstacle, ensure_obstacles
from .sim_2d import SoftArmSim2D


def _flatten_query_points(query_points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(query_points, dtype=float)
    if points.shape == (2,):
        return points.reshape(1, 2), ()
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("query_points must have shape (2,) or (..., 2).")
    return points.reshape(-1, 2), points.shape[:-1]


def _reshape_scalar_field(values: np.ndarray, output_shape: tuple[int, ...]) -> float | np.ndarray:
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    if not output_shape:
        return float(flat_values[0])
    return flat_values.reshape(output_shape)


def _reshape_vector_field(values: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    vectors = np.asarray(values, dtype=float).reshape(-1, 2)
    if not output_shape:
        return vectors[0].copy()
    return vectors.reshape(output_shape + (2,))


def compute_tip_space_attractive_potential(
    query_points: np.ndarray,
    *,
    target: np.ndarray,
    attractive_gain: float,
) -> float | np.ndarray:
    flat_points, output_shape = _flatten_query_points(query_points)
    target = np.asarray(target, dtype=float).reshape(2)
    deltas = flat_points - target[None, :]
    potential = 0.5 * float(attractive_gain) * np.sum(deltas * deltas, axis=1)
    return _reshape_scalar_field(potential, output_shape)


def compute_tip_space_repulsive_potential(
    query_points: np.ndarray,
    *,
    obstacles: list[CircleObstacle | tuple[float, float, float]],
    repulsive_gain: float,
    safe_threshold: float,
    min_clearance: float,
) -> float | np.ndarray:
    flat_points, output_shape = _flatten_query_points(query_points)
    normalized_obstacles = ensure_obstacles(obstacles or [])
    if not normalized_obstacles:
        return _reshape_scalar_field(np.zeros(flat_points.shape[0], dtype=float), output_shape)

    safe_threshold = max(float(safe_threshold), float(min_clearance), 1e-8)
    min_clearance = max(float(min_clearance), 1e-8)
    potential = np.zeros(flat_points.shape[0], dtype=float)

    for obstacle in normalized_obstacles:
        deltas = flat_points - obstacle.center[None, :]
        distances = np.linalg.norm(deltas, axis=1)
        clearances = distances - float(obstacle.radius)
        active_mask = clearances < safe_threshold
        if not np.any(active_mask):
            continue

        effective_clearances = np.maximum(clearances[active_mask], min_clearance)
        inverse_gap = (1.0 / effective_clearances) - (1.0 / safe_threshold)
        potential[active_mask] += 0.5 * float(repulsive_gain) * (inverse_gap * inverse_gap)

    return _reshape_scalar_field(potential, output_shape)


def compute_tip_space_simplified_apf_potential(
    query_points: np.ndarray,
    *,
    target: np.ndarray,
    obstacles: list[CircleObstacle | tuple[float, float, float]],
    attractive_gain: float,
    repulsive_gain: float,
    safe_threshold: float,
    min_clearance: float,
) -> float | np.ndarray:
    """Return a tip-space simplified APF potential for visualization only.

    This helper evaluates the APF directly in workspace tip coordinates. It is
    intended for debugging figures and paper illustrations, and is not the full
    continuum-arm configuration-space potential optimized by the controller.
    """

    attractive = np.asarray(
        compute_tip_space_attractive_potential(
            query_points,
            target=target,
            attractive_gain=attractive_gain,
        ),
        dtype=float,
    )
    repulsive = np.asarray(
        compute_tip_space_repulsive_potential(
            query_points,
            obstacles=obstacles,
            repulsive_gain=repulsive_gain,
            safe_threshold=safe_threshold,
            min_clearance=min_clearance,
        ),
        dtype=float,
    )
    if attractive.ndim == 0 and repulsive.ndim == 0:
        return float(attractive + repulsive)
    return attractive + repulsive


def compute_tip_space_simplified_apf_force(
    query_points: np.ndarray,
    *,
    target: np.ndarray,
    obstacles: list[CircleObstacle | tuple[float, float, float]],
    attractive_gain: float,
    repulsive_gain: float,
    safe_threshold: float,
    min_clearance: float,
) -> np.ndarray:
    """Return the negative potential gradient in tip space.

    The returned force field is consistent with the simplified visualization
    potential above. Inside the `min_clearance` clamp band the repulsive
    potential is flattened for numerical stability, so the repulsive gradient is
    suppressed there as a visualization choice.
    """

    flat_points, output_shape = _flatten_query_points(query_points)
    target = np.asarray(target, dtype=float).reshape(2)
    normalized_obstacles = ensure_obstacles(obstacles or [])
    safe_threshold = max(float(safe_threshold), float(min_clearance), 1e-8)
    min_clearance = max(float(min_clearance), 1e-8)

    total_force = float(attractive_gain) * (target[None, :] - flat_points)

    for obstacle in normalized_obstacles:
        deltas = flat_points - obstacle.center[None, :]
        distances = np.linalg.norm(deltas, axis=1)
        clearances = distances - float(obstacle.radius)
        active_mask = (clearances < safe_threshold) & (clearances > min_clearance)
        if not np.any(active_mask):
            continue

        active_deltas = deltas[active_mask]
        active_distances = np.maximum(distances[active_mask], 1e-8)
        directions = active_deltas / active_distances[:, None]
        effective_clearances = np.maximum(clearances[active_mask], min_clearance)
        magnitudes = float(repulsive_gain) * (
            (1.0 / effective_clearances) - (1.0 / safe_threshold)
        ) / np.maximum(effective_clearances * effective_clearances, 1e-8)
        total_force[active_mask] += magnitudes[:, None] * directions

    return _reshape_vector_field(total_force, output_shape)


def sample_tip_space_simplified_apf_grid(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int | tuple[int, int],
    *,
    target: np.ndarray,
    obstacles: list[CircleObstacle | tuple[float, float, float]],
    attractive_gain: float,
    repulsive_gain: float,
    safe_threshold: float,
    min_clearance: float,
    return_force_field: bool = False,
    visualization_clip_quantile: float | None = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a tip-space simplified APF grid for background visualization.

    The optional quantile clip is applied only to the displayed potential values
    so obstacle-adjacent peaks do not dominate the whole colormap. It does not
    redefine the underlying simplified APF.
    """

    if isinstance(resolution, tuple):
        nx = max(int(resolution[0]), 2)
        ny = max(int(resolution[1]), 2)
    else:
        nx = max(int(resolution), 2)
        ny = max(int(resolution), 2)

    xs = np.linspace(float(xlim[0]), float(xlim[1]), nx, dtype=float)
    ys = np.linspace(float(ylim[0]), float(ylim[1]), ny, dtype=float)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    query_points = np.stack((X, Y), axis=-1)

    potential = np.asarray(
        compute_tip_space_simplified_apf_potential(
            query_points,
            target=target,
            obstacles=obstacles,
            attractive_gain=attractive_gain,
            repulsive_gain=repulsive_gain,
            safe_threshold=safe_threshold,
            min_clearance=min_clearance,
        ),
        dtype=float,
    )
    display_potential = potential.copy()
    if visualization_clip_quantile is not None:
        clip_quantile = float(visualization_clip_quantile)
        if 0.0 < clip_quantile < 100.0:
            finite_values = display_potential[np.isfinite(display_potential)]
            if finite_values.size:
                clip_upper = float(np.quantile(finite_values, clip_quantile / 100.0))
                if np.isfinite(clip_upper):
                    display_potential = np.clip(display_potential, None, clip_upper)

    if not return_force_field:
        return X, Y, display_potential

    force_field = np.asarray(
        compute_tip_space_simplified_apf_force(
            query_points,
            target=target,
            obstacles=obstacles,
            attractive_gain=attractive_gain,
            repulsive_gain=repulsive_gain,
            safe_threshold=safe_threshold,
            min_clearance=min_clearance,
        ),
        dtype=float,
    )
    return X, Y, display_potential, force_field[..., 0], force_field[..., 1]


@dataclass(frozen=True)
class APFController2DConfig:
    attractive_gain: float = 10.35
    repulsive_gain: float = 0.10
    safe_threshold: float = 3.0
    action_gain: float = 0.5
    finite_difference_epsilon: float = 1e-4
    backbone_samples_per_segment: int = 9
    min_clearance: float = 1e-2
    max_repulsive_force: float = 4.0
    max_total_force: float = 8.0
    base_movement_weight: float = 10.0
    damping_lambda: float = 0.01
    enable_tangential_escape: bool = True
    tangential_escape_force: float = 0.1
    tangential_escape_weight: float = 0.0


class APFController2D:
    def __init__(
        self,
        *,
        control_config: TendonControl2DConfig | None = None,
        config: APFController2DConfig | None = None,
    ):
        self.control_config = control_config or TendonControl2DConfig()
        self.config = config or APFController2DConfig()
        self.last_control_info: dict[str, object] = {}

    def reset(self) -> None:
        self.last_control_info = {}

    def compute_attractive_force(self, sim: SoftArmSim2D) -> np.ndarray:
        tip = sim.model.tip_position
        target = sim.target
        return float(self.config.attractive_gain) * (target - tip)

    def _sample_backbone(self, sim: SoftArmSim2D) -> tuple[np.ndarray, np.ndarray]:
        segments = sim.model.observed_chain.segments
        if not segments:
            return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)

        total_length = max(float(np.sum([segment.length for segment in segments])), 1e-8)
        base_radius = float(sim.model.config.radius_base)
        tip_radius = (
            base_radius
            if sim.model.config.radius_tip is None
            else float(sim.model.config.radius_tip)
        )
        points: list[np.ndarray] = []
        radii: list[float] = []
        prefix = 0.0
        samples_per_segment = max(int(self.config.backbone_samples_per_segment), 3)

        for segment_index, segment in enumerate(segments):
            fractions = np.linspace(0.0, 1.0, samples_per_segment, dtype=float)
            if segment_index:
                fractions = fractions[1:]
            segment_points, _ = segment.pose_at_fractions(fractions)
            for fraction, point in zip(fractions, segment_points):
                arclength = prefix + float(fraction) * float(segment.length)
                global_fraction = float(np.clip(arclength / total_length, 0.0, 1.0))
                radius = base_radius - global_fraction * (base_radius - tip_radius)
                points.append(point.copy())
                radii.append(float(radius))
            prefix += float(segment.length)

        return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)

    def _copy_repulsive_contacts(self, repulsive_contacts: list[dict[str, object]]) -> list[dict[str, object]]:
        copied_contacts: list[dict[str, object]] = []
        for contact in repulsive_contacts:
            copied_contact: dict[str, object] = {}
            for key, value in contact.items():
                copied_contact[key] = value.copy() if isinstance(value, np.ndarray) else value
            copied_contacts.append(copied_contact)
        return copied_contacts

    def _sum_repulsive_contact_forces(self, repulsive_contacts: list[dict[str, object]]) -> np.ndarray:
        total_force = np.zeros(2, dtype=float)
        for contact in repulsive_contacts:
            total_force += np.asarray(contact["force"], dtype=float)
        return total_force

    def compute_repulsive_contacts(self, sim: SoftArmSim2D) -> list[dict[str, object]]:
        points, radii = self._sample_backbone(sim)
        if points.size == 0 or not sim.obstacles:
            return []

        safe_threshold = max(float(self.config.safe_threshold), float(self.config.min_clearance))
        fallback_direction = normalize(sim.target - sim.model.tip_position)
        repulsive_contacts: list[dict[str, object]] = []

        for obstacle_index, obstacle in enumerate(sim.obstacles):
            deltas = points - obstacle.center[None, :]
            distances = np.linalg.norm(deltas, axis=1)
            clearances = distances - (float(obstacle.radius) + radii)
            closest_index = int(np.argmin(clearances))
            clearance = float(clearances[closest_index])
            if clearance >= safe_threshold:
                continue

            direction = normalize(deltas[closest_index], fallback=fallback_direction)
            effective_clearance = max(clearance, float(self.config.min_clearance))
            magnitude = float(self.config.repulsive_gain) * (
                (1.0 / effective_clearance) - (1.0 / safe_threshold)
            ) / max(effective_clearance * effective_clearance, 1e-8)
            magnitude = min(magnitude, float(self.config.max_repulsive_force))
            repulsive_contacts.append(
                {
                    "obstacle_index": obstacle_index,
                    "sample_index": closest_index,
                    "closest_point": points[closest_index].copy(),
                    "clearance": clearance,
                    "effective_clearance": float(effective_clearance),
                    "arm_radius": float(radii[closest_index]),
                    "force": (magnitude * direction).copy(),
                    "force_magnitude": float(magnitude),
                    "force_direction": direction.copy(),
                    "obstacle_center": obstacle.center.copy(),
                    "obstacle_radius": float(obstacle.radius),
                }
            )

        return repulsive_contacts

    def compute_repulsive_force(self, sim: SoftArmSim2D) -> np.ndarray:
        repulsive_contacts = self.compute_repulsive_contacts(sim)
        total_force = self._sum_repulsive_contact_forces(repulsive_contacts)
        self.last_control_info["repulsive_contacts"] = self._copy_repulsive_contacts(repulsive_contacts)
        return total_force

    def _restore_probe_state(
        self,
        sim: SoftArmSim2D,
        state: dict[str, object],
        *,
        base_x: float | None = None,
        tendon_lengths: np.ndarray | None = None,
    ) -> None:
        probe_state = dict(state)
        if base_x is not None:
            probe_state["base_x"] = float(base_x)
        if tendon_lengths is not None:
            probe_state["free_tendon_lengths"] = np.asarray(tendon_lengths, dtype=float).copy()
        sim.model.restore(probe_state)

    def _probe_tip_position(
        self,
        sim: SoftArmSim2D,
        state: dict[str, object],
        *,
        base_x: float | None = None,
        tendon_lengths: np.ndarray | None = None,
    ) -> np.ndarray:
        self._restore_probe_state(
            sim,
            state,
            base_x=base_x,
            tendon_lengths=tendon_lengths,
        )
        return sim.model.tip_position.copy()

    def _probe_backbone_point_position(
        self,
        sim: SoftArmSim2D,
        state: dict[str, object],
        *,
        sample_index: int,
        base_x: float | None = None,
        tendon_lengths: np.ndarray | None = None,
    ) -> np.ndarray:
        self._restore_probe_state(
            sim,
            state,
            base_x=base_x,
            tendon_lengths=tendon_lengths,
        )
        points, _ = self._sample_backbone(sim)
        if not 0 <= int(sample_index) < len(points):
            raise IndexError(f"sample_index={sample_index} is out of bounds for {len(points)} backbone samples.")
        return points[int(sample_index)].copy()

    def _compute_workspace_point_jacobian(
        self,
        sim: SoftArmSim2D,
        *,
        state: dict[str, object],
        baseline_position: np.ndarray,
        probe_position,
    ) -> np.ndarray:
        model = sim.model
        base_x = float(model.base_x)
        baseline_tendon_lengths = model.free_tendon_lengths.copy()
        tendon_lengths = baseline_tendon_lengths.reshape(-1).copy()
        base_lower, base_upper = map(float, model.config.base_x_limits)
        tendon_lower, tendon_upper = map(float, model.config.tendon_length_limits)
        epsilon = max(float(self.config.finite_difference_epsilon), 1e-6)
        jacobian = np.zeros((2, 1 + tendon_lengths.size), dtype=float)

        try:
            plus_step = min(epsilon, base_upper - base_x)
            minus_step = min(epsilon, base_x - base_lower)

            if plus_step > 1e-12 and minus_step > 1e-12:
                tip_plus = probe_position(
                    sim,
                    state,
                    base_x=base_x + plus_step,
                    tendon_lengths=baseline_tendon_lengths,
                )
                tip_minus = probe_position(
                    sim,
                    state,
                    base_x=base_x - minus_step,
                    tendon_lengths=baseline_tendon_lengths,
                )
                jacobian[:, 0] = (tip_plus - tip_minus) / (plus_step + minus_step)
            elif plus_step > 1e-12:
                tip_plus = probe_position(
                    sim,
                    state,
                    base_x=base_x + plus_step,
                    tendon_lengths=baseline_tendon_lengths,
                )
                jacobian[:, 0] = (tip_plus - baseline_position) / plus_step
            elif minus_step > 1e-12:
                tip_minus = probe_position(
                    sim,
                    state,
                    base_x=base_x - minus_step,
                    tendon_lengths=baseline_tendon_lengths,
                )
                jacobian[:, 0] = (baseline_position - tip_minus) / minus_step

            for index in range(tendon_lengths.size):
                column_index = 1 + index
                plus_step = min(epsilon, tendon_upper - float(tendon_lengths[index]))
                minus_step = min(epsilon, float(tendon_lengths[index]) - tendon_lower)

                if plus_step > 1e-12 and minus_step > 1e-12:
                    plus = tendon_lengths.copy()
                    minus = tendon_lengths.copy()
                    plus[index] += plus_step
                    minus[index] -= minus_step
                    tip_plus = probe_position(
                        sim,
                        state,
                        base_x=base_x,
                        tendon_lengths=plus.reshape(model.free_tendon_lengths.shape),
                    )
                    tip_minus = probe_position(
                        sim,
                        state,
                        base_x=base_x,
                        tendon_lengths=minus.reshape(model.free_tendon_lengths.shape),
                    )
                    jacobian[:, column_index] = (tip_plus - tip_minus) / (plus_step + minus_step)
                elif plus_step > 1e-12:
                    plus = tendon_lengths.copy()
                    plus[index] += plus_step
                    tip_plus = probe_position(
                        sim,
                        state,
                        base_x=base_x,
                        tendon_lengths=plus.reshape(model.free_tendon_lengths.shape),
                    )
                    jacobian[:, column_index] = (tip_plus - baseline_position) / plus_step
                elif minus_step > 1e-12:
                    minus = tendon_lengths.copy()
                    minus[index] -= minus_step
                    tip_minus = probe_position(
                        sim,
                        state,
                        base_x=base_x,
                        tendon_lengths=minus.reshape(model.free_tendon_lengths.shape),
                    )
                    jacobian[:, column_index] = (baseline_position - tip_minus) / minus_step
        finally:
            model.restore(state)

        return jacobian

    def compute_kinematic_jacobian(self, sim: SoftArmSim2D) -> np.ndarray:
        model = sim.model
        state = model.snapshot()
        baseline_tip = model.tip_position.copy()
        return self._compute_workspace_point_jacobian(
            sim,
            state=state,
            baseline_position=baseline_tip,
            probe_position=self._probe_tip_position,
        )

    def compute_backbone_point_jacobian(self, sim: SoftArmSim2D, *, sample_index: int) -> np.ndarray:
        model = sim.model
        state = model.snapshot()
        points, _ = self._sample_backbone(sim)
        if not 0 <= int(sample_index) < len(points):
            raise IndexError(f"sample_index={sample_index} is out of bounds for {len(points)} backbone samples.")
        baseline_point = points[int(sample_index)].copy()
        return self._compute_workspace_point_jacobian(
            sim,
            state=state,
            baseline_position=baseline_point,
            probe_position=lambda sim_, state_, *, base_x=None, tendon_lengths=None: self._probe_backbone_point_position(
                sim_,
                state_,
                sample_index=int(sample_index),
                base_x=base_x,
                tendon_lengths=tendon_lengths,
            ),
        )

    def _compute_total_force(
        self,
        *,
        attractive_force: np.ndarray,
        repulsive_force: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        raw_total_force = np.asarray(attractive_force, dtype=float) + np.asarray(repulsive_force, dtype=float)
        total_force = raw_total_force.copy()
        total_force_scale = 1.0
        force_norm = float(np.linalg.norm(total_force))
        max_total_force = float(self.config.max_total_force)
        if force_norm > max_total_force > 0.0:
            total_force_scale = max_total_force / force_norm
            total_force *= total_force_scale
        return raw_total_force, total_force, float(total_force_scale)

    def _compute_contact_tangential_forces(
        self,
        contact: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        normal_direction = normalize(
            np.asarray(contact["closest_point"], dtype=float) - np.asarray(contact["obstacle_center"], dtype=float),
            fallback=np.asarray(contact["force_direction"], dtype=float),
        )
        tangent_left = np.array([-normal_direction[1], normal_direction[0]], dtype=float)
        tangent_right = np.array([normal_direction[1], -normal_direction[0]], dtype=float)
        tangential_force = max(float(self.config.tangential_escape_force), 0.0)
        return tangential_force * tangent_left, tangential_force * tangent_right

    def _base_control_enabled(self) -> bool:
        return float(self.control_config.base_delta_scale) > 1e-8

    def _build_inverse_weight_vector(self, control_count: int) -> np.ndarray:
        inverse_weights = np.ones(control_count, dtype=float)
        if control_count == 0:
            return inverse_weights
        inverse_weights[0] = max(float(self.config.base_movement_weight), 0.0)
        if not self._base_control_enabled():
            inverse_weights[0] = 0.0
        return inverse_weights

    def _solve_joint_deltas_from_force(
        self,
        sim: SoftArmSim2D,
        *,
        jacobian: np.ndarray,
        cartesian_force: np.ndarray,
    ) -> np.ndarray:
        del sim
        inverse_weights = self._build_inverse_weight_vector(jacobian.shape[1])
        weighted_jacobian = jacobian * inverse_weights[None, :]
        damping = max(float(self.config.damping_lambda), 0.0)
        system_matrix = weighted_jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0], dtype=float)

        try:
            cartesian_response = np.linalg.solve(system_matrix, np.asarray(cartesian_force, dtype=float))
        except np.linalg.LinAlgError:
            cartesian_response = np.linalg.pinv(system_matrix) @ np.asarray(cartesian_force, dtype=float)

        joint_deltas = (inverse_weights[:, None] * jacobian.T) @ cartesian_response.reshape(-1, 1)
        return float(self.config.action_gain) * joint_deltas.reshape(-1)

    def _clip_joint_deltas_to_model_limits(self, sim: SoftArmSim2D, joint_deltas: np.ndarray) -> np.ndarray:
        clipped = np.asarray(joint_deltas, dtype=float).reshape(-1).copy()
        if clipped.size != 1 + (sim.model.num_segments * 2):
            raise ValueError(
                "joint_deltas must contain one base command followed by two tendon commands per segment."
            )

        model = sim.model
        base_lower, base_upper = map(float, model.config.base_x_limits)
        tendon_lower, tendon_upper = map(float, model.config.tendon_length_limits)

        if self._base_control_enabled():
            base_delta_lower = base_lower - float(model.base_x)
            base_delta_upper = base_upper - float(model.base_x)
            if not self.control_config.allow_negative_base_delta:
                base_delta_lower = max(base_delta_lower, 0.0)
            clipped[0] = float(np.clip(clipped[0], base_delta_lower, base_delta_upper))
        else:
            clipped[0] = 0.0

        tendon_lengths = model.free_tendon_lengths.reshape(-1)
        lower_deltas = tendon_lower - tendon_lengths
        upper_deltas = tendon_upper - tendon_lengths
        clipped[1:] = np.clip(clipped[1:], lower_deltas, upper_deltas)
        return clipped

    def compute_desired_joint_deltas(self, sim: SoftArmSim2D) -> np.ndarray:
        attractive_force = self.compute_attractive_force(sim)
        repulsive_contacts = self.compute_repulsive_contacts(sim)
        repulsive_force = self._sum_repulsive_contact_forces(repulsive_contacts)
        raw_total_force, total_force, total_force_scale = self._compute_total_force(
            attractive_force=attractive_force,
            repulsive_force=repulsive_force,
        )

        tip_jacobian = self.compute_kinematic_jacobian(sim)
        attractive_joint_deltas = self._solve_joint_deltas_from_force(
            sim,
            jacobian=tip_jacobian,
            cartesian_force=attractive_force,
        )

        # This is a more faithful kinematic APF mapping for a continuum arm:
        # attraction is tracked at the tip, while each obstacle repulsion is
        # mapped from the nominal nearest backbone point with a fixed local
        # Jacobian. It is still a kinematic approximation, not full dynamics or
        # optimization-based contact control.
        #
        # The tangential escape term below is an additional multi-contact
        # heuristic layered on top of this kinematic APF. For every active
        # repulsive contact we keep the normal repulsion, and simultaneously add
        # left/right tangential guidance terms at that same local Jacobian. The
        # goal is to help the soft arm slide along obstacle boundaries when
        # several local danger points are active at once, rather than only being
        # pushed outward by normal repulsion. It is still not full dynamics, not
        # globally optimal planning, and not optimization-based contact planning.
        repulsive_joint_deltas_per_contact: list[np.ndarray] = []
        tangential_forces_left_per_contact: list[np.ndarray] = []
        tangential_forces_right_per_contact: list[np.ndarray] = []
        tangential_joint_deltas_left_per_contact: list[np.ndarray] = []
        tangential_joint_deltas_right_per_contact: list[np.ndarray] = []
        used_contact_jacobians: list[np.ndarray] = []
        tangential_escape_active = bool(self.config.enable_tangential_escape) and bool(repulsive_contacts)
        tangential_escape_weight = max(float(self.config.tangential_escape_weight), 0.0)
        for contact in repulsive_contacts:
            contact_jacobian = self.compute_backbone_point_jacobian(
                sim,
                sample_index=int(contact["sample_index"]),
            )
            contact_joint_deltas = self._solve_joint_deltas_from_force(
                sim,
                jacobian=contact_jacobian,
                cartesian_force=np.asarray(contact["force"], dtype=float),
            )
            used_contact_jacobians.append(contact_jacobian.copy())
            repulsive_joint_deltas_per_contact.append(contact_joint_deltas.copy())
            if tangential_escape_active:
                tangential_force_left, tangential_force_right = self._compute_contact_tangential_forces(contact)
                tangential_joint_deltas_left_contact = self._solve_joint_deltas_from_force(
                    sim,
                    jacobian=contact_jacobian,
                    cartesian_force=tangential_force_left,
                )
                tangential_joint_deltas_right_contact = self._solve_joint_deltas_from_force(
                    sim,
                    jacobian=contact_jacobian,
                    cartesian_force=tangential_force_right,
                )
                tangential_forces_left_per_contact.append(tangential_force_left.copy())
                tangential_forces_right_per_contact.append(tangential_force_right.copy())
                tangential_joint_deltas_left_per_contact.append(tangential_joint_deltas_left_contact.copy())
                tangential_joint_deltas_right_per_contact.append(tangential_joint_deltas_right_contact.copy())

        repulsive_joint_deltas = np.zeros_like(attractive_joint_deltas)
        for contact_joint_deltas in repulsive_joint_deltas_per_contact:
            repulsive_joint_deltas += contact_joint_deltas

        tangential_joint_deltas_left = np.zeros_like(attractive_joint_deltas)
        for tangential_joint_deltas in tangential_joint_deltas_left_per_contact:
            tangential_joint_deltas_left += tangential_joint_deltas

        tangential_joint_deltas_right = np.zeros_like(attractive_joint_deltas)
        for tangential_joint_deltas in tangential_joint_deltas_right_per_contact:
            tangential_joint_deltas_right += tangential_joint_deltas

        combined_joint_deltas_before_force_clip = attractive_joint_deltas + repulsive_joint_deltas
        raw_joint_deltas = total_force_scale * combined_joint_deltas_before_force_clip
        joint_deltas = self._clip_joint_deltas_to_model_limits(sim, raw_joint_deltas)
        tendon_deltas = joint_deltas[1:].reshape(sim.model.free_tendon_lengths.shape)
        tangential_candidate_left: np.ndarray | None = None
        tangential_candidate_right: np.ndarray | None = None
        if tangential_escape_active:
            tangential_candidate_left = self._clip_joint_deltas_to_model_limits(
                sim,
                total_force_scale
                * (
                    combined_joint_deltas_before_force_clip
                    + tangential_escape_weight * tangential_joint_deltas_left
                ),
            )
            tangential_candidate_right = self._clip_joint_deltas_to_model_limits(
                sim,
                total_force_scale
                * (
                    combined_joint_deltas_before_force_clip
                    + tangential_escape_weight * tangential_joint_deltas_right
                ),
            )

        self.last_control_info = {
            **self.last_control_info,
            "attractive_force": attractive_force.copy(),
            "repulsive_force": repulsive_force.copy(),
            "repulsive_contacts": self._copy_repulsive_contacts(repulsive_contacts),
            "raw_total_force": raw_total_force.copy(),
            "total_force": total_force.copy(),
            "kinematic_jacobian": tip_jacobian.copy(),
            "full_kinematic_jacobian": tip_jacobian.copy(),
            "used_tip_jacobian": tip_jacobian.copy(),
            "used_contact_jacobians": [jacobian.copy() for jacobian in used_contact_jacobians],
            "inverse_weight_vector": self._build_inverse_weight_vector(tip_jacobian.shape[1]).copy(),
            "attractive_joint_deltas": attractive_joint_deltas.copy(),
            "repulsive_joint_deltas_per_contact": [deltas.copy() for deltas in repulsive_joint_deltas_per_contact],
            "tangential_forces_left_per_contact": [force.copy() for force in tangential_forces_left_per_contact],
            "tangential_forces_right_per_contact": [force.copy() for force in tangential_forces_right_per_contact],
            "tangential_joint_deltas_left_per_contact": [
                deltas.copy() for deltas in tangential_joint_deltas_left_per_contact
            ],
            "tangential_joint_deltas_right_per_contact": [
                deltas.copy() for deltas in tangential_joint_deltas_right_per_contact
            ],
            "repulsive_joint_deltas": repulsive_joint_deltas.copy(),
            "tangential_joint_deltas_left": tangential_joint_deltas_left.copy(),
            "tangential_joint_deltas_right": tangential_joint_deltas_right.copy(),
            "tangential_candidate_left": None if tangential_candidate_left is None else tangential_candidate_left.copy(),
            "tangential_candidate_right": None if tangential_candidate_right is None else tangential_candidate_right.copy(),
            "tangential_escape_active": bool(tangential_escape_active),
            "combined_joint_deltas_before_force_clip": combined_joint_deltas_before_force_clip.copy(),
            "combined_joint_deltas_before_clip": raw_joint_deltas.copy(),
            "raw_desired_joint_deltas": raw_joint_deltas.copy(),
            "desired_joint_deltas": joint_deltas.copy(),
            "desired_base_delta": float(joint_deltas[0]),
            "desired_tendon_deltas": tendon_deltas.copy(),
            "distance_to_target": float(sim.distance_to_target()),
            "min_clearance": float(sim.contact_summary.min_clearance),
            "base_movement_weight": float(self.config.base_movement_weight),
            "damping_lambda": float(self.config.damping_lambda),
            "total_force_scale": float(total_force_scale),
            "tangential_escape_force": float(self.config.tangential_escape_force),
            "tangential_escape_weight": float(self.config.tangential_escape_weight),
        }
        return joint_deltas

    def compute_desired_tendon_deltas(self, sim: SoftArmSim2D) -> np.ndarray:
        joint_deltas = self.compute_desired_joint_deltas(sim)
        return joint_deltas[1:].reshape(sim.model.free_tendon_lengths.shape)

    def _joint_deltas_to_action(self, sim: SoftArmSim2D, joint_deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped_joint_deltas = self._clip_joint_deltas_to_model_limits(sim, joint_deltas)
        action = np.zeros(self.control_config.action_dim(sim.model.num_segments), dtype=np.float32)

        base_scale = float(self.control_config.base_delta_scale)
        if base_scale > 1e-8:
            action[0] = np.float32(clipped_joint_deltas[0] / base_scale)

        tendon_scale = float(self.control_config.tendon_delta_scale)
        if tendon_scale > 1e-8:
            action[1:] = (
                clipped_joint_deltas[1:].reshape(sim.model.free_tendon_lengths.shape).reshape(-1) / tendon_scale
            ).astype(np.float32, copy=False)

        action = np.clip(action, self.control_config.action_low, self.control_config.action_high)
        clipped_action, base_delta, tendon_deltas = self.control_config.decode_action(action, sim.model.num_segments)
        applied_joint_deltas = np.concatenate(
            (
                np.array([base_delta], dtype=float),
                tendon_deltas.reshape(-1).astype(float, copy=False),
            )
        )
        applied_joint_deltas = self._clip_joint_deltas_to_model_limits(sim, applied_joint_deltas)
        final_action = np.zeros_like(clipped_action, dtype=np.float32)
        if base_scale > 1e-8:
            final_action[0] = np.float32(applied_joint_deltas[0] / base_scale)
        if tendon_scale > 1e-8:
            final_action[1:] = (
                applied_joint_deltas[1:].reshape(sim.model.free_tendon_lengths.shape).reshape(-1) / tendon_scale
            ).astype(np.float32, copy=False)
        final_action = np.clip(final_action, self.control_config.action_low, self.control_config.action_high)
        return final_action.astype(np.float32, copy=False), applied_joint_deltas

    def _preview_joint_deltas(self, sim: SoftArmSim2D, joint_deltas: np.ndarray) -> dict[str, object]:
        state = sim.snapshot()
        try:
            sim.step(
                base_delta=float(joint_deltas[0]),
                tendon_length_deltas=joint_deltas[1:].reshape(sim.model.free_tendon_lengths.shape),
            )
            return {
                "action_blocked": bool(sim.model.action_blocked),
                "collision_count": int(len(sim.model.collision_events)),
                "min_clearance": float(sim.contact_summary.min_clearance),
                "distance_to_target": float(sim.distance_to_target()),
            }
        finally:
            sim.restore(state)

    def _is_preview_safe(self, preview: dict[str, object]) -> bool:
        return (
            not bool(preview["action_blocked"])
            and int(preview["collision_count"]) == 0
            and float(preview["min_clearance"]) > 0.0
        )

    def _record_selected_action(
        self,
        sim: SoftArmSim2D,
        *,
        action: np.ndarray,
        applied_joint_deltas: np.ndarray,
        selected_label: str,
        selected_scale: float,
        preview: dict[str, object],
    ) -> None:
        self.last_control_info["action"] = action.copy()
        self.last_control_info["applied_base_delta"] = float(applied_joint_deltas[0])
        self.last_control_info["applied_tendon_deltas"] = applied_joint_deltas[1:].reshape(
            sim.model.free_tendon_lengths.shape
        )
        self.last_control_info["selected_candidate"] = selected_label
        self.last_control_info["selected_action_scale"] = float(selected_scale)
        self.last_control_info["preview"] = dict(preview)

    def plan_action(self, sim: SoftArmSim2D) -> np.ndarray:
        desired_joint_deltas = self.compute_desired_joint_deltas(sim)
        candidate_joint_deltas: list[tuple[str, np.ndarray]] = []
        tangential_escape_active = bool(self.last_control_info.get("tangential_escape_active", False))

        if tangential_escape_active:
            tangential_candidate_left = self.last_control_info.get("tangential_candidate_left")
            tangential_candidate_right = self.last_control_info.get("tangential_candidate_right")
            if isinstance(tangential_candidate_left, np.ndarray):
                candidate_joint_deltas.append(("tangent_left", tangential_candidate_left.copy()))
            if isinstance(tangential_candidate_right, np.ndarray):
                candidate_joint_deltas.append(("tangent_right", tangential_candidate_right.copy()))

        candidate_joint_deltas.append(("total_force", desired_joint_deltas.copy()))

        repulsive_joint_deltas = np.asarray(
            self.last_control_info.get("repulsive_joint_deltas", np.zeros_like(desired_joint_deltas)),
            dtype=float,
        )
        if float(np.linalg.norm(repulsive_joint_deltas)) > 0.0:
            repulsive_joint_deltas = self._clip_joint_deltas_to_model_limits(
                sim,
                repulsive_joint_deltas,
            )
            candidate_joint_deltas.append(("repulsive_only", repulsive_joint_deltas))

        preview_scales = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
        selected_label = "total_force"
        selected_scale = 0.0
        selected_preview = {
            "action_blocked": False,
            "collision_count": 0,
            "min_clearance": float(sim.contact_summary.min_clearance),
            "distance_to_target": float(sim.distance_to_target()),
        }
        action = np.zeros(self.control_config.action_dim(sim.model.num_segments), dtype=np.float32)
        applied_joint_deltas = np.zeros(1 + (sim.model.num_segments * 2), dtype=float)
        best_safe_result: dict[str, object] | None = None
        rank_safe_candidates = tangential_escape_active and len(candidate_joint_deltas) > 1

        for candidate_index, (label, candidate_joint) in enumerate(candidate_joint_deltas):
            for scale in preview_scales:
                scaled_joint_deltas = self._clip_joint_deltas_to_model_limits(sim, float(scale) * candidate_joint)
                candidate_action, candidate_applied_joint_deltas = self._joint_deltas_to_action(sim, scaled_joint_deltas)
                preview = self._preview_joint_deltas(sim, candidate_applied_joint_deltas)
                if self._is_preview_safe(preview):
                    if rank_safe_candidates:
                        preview_score = (
                            float(preview["distance_to_target"]),
                            -float(preview["min_clearance"]),
                            -float(scale),
                            int(candidate_index),
                        )
                        if best_safe_result is None or preview_score < best_safe_result["score"]:
                            best_safe_result = {
                                "score": preview_score,
                                "action": candidate_action.copy(),
                                "applied_joint_deltas": candidate_applied_joint_deltas.copy(),
                                "selected_label": label,
                                "selected_scale": float(scale),
                                "preview": dict(preview),
                            }
                        continue
                    action = candidate_action
                    applied_joint_deltas = candidate_applied_joint_deltas
                    selected_label = label
                    selected_scale = float(scale)
                    selected_preview = preview
                    self._record_selected_action(
                        sim,
                        action=action,
                        applied_joint_deltas=applied_joint_deltas,
                        selected_label=selected_label,
                        selected_scale=selected_scale,
                        preview=selected_preview,
                    )
                    return action

        if best_safe_result is not None:
            action = np.asarray(best_safe_result["action"], dtype=np.float32)
            applied_joint_deltas = np.asarray(best_safe_result["applied_joint_deltas"], dtype=float)
            selected_label = str(best_safe_result["selected_label"])
            selected_scale = float(best_safe_result["selected_scale"])
            selected_preview = dict(best_safe_result["preview"])
            self._record_selected_action(
                sim,
                action=action,
                applied_joint_deltas=applied_joint_deltas,
                selected_label=selected_label,
                selected_scale=selected_scale,
                preview=selected_preview,
            )
            return action

        zero_action, zero_joint_deltas = self._joint_deltas_to_action(
            sim,
            np.zeros(1 + (sim.model.num_segments * 2), dtype=float),
        )
        zero_preview = self._preview_joint_deltas(sim, zero_joint_deltas)
        self._record_selected_action(
            sim,
            action=zero_action,
            applied_joint_deltas=zero_joint_deltas,
            selected_label="zero_fallback",
            selected_scale=0.0,
            preview=zero_preview,
        )
        return zero_action

    def act(self, sim: SoftArmSim2D) -> np.ndarray:
        return self.plan_action(sim)
