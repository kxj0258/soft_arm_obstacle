from __future__ import annotations

import numpy as np
import pytest

from tasks.arm2d.model_2d import ArmModel2D, ArmModel2DConfig
from tasks.arm2d.physics_2d import CircleObstacle


EPSILON = 1e-6


def _assert_polyline_stays_outside_obstacle(model: ArmModel2D, obstacle: CircleObstacle, *, epsilon: float = EPSILON) -> None:
    polyline = model.observed_polyline()
    distances = np.linalg.norm(polyline - obstacle.center[None, :], axis=1)
    assert np.all(distances + epsilon >= obstacle.radius), (
        f"Observed polyline penetrated obstacle: min_distance={float(np.min(distances)):.6f}, "
        f"radius={float(obstacle.radius):.6f}"
    )


@pytest.mark.parametrize(
    ("name", "config", "obstacle", "base_x", "kappas", "base_delta", "tendon_deltas", "min_boundary_samples"),
    [
        (
            "single_point_front_block",
            ArmModel2DConfig(
                segment_lengths=(10.0, 10.0),
                base_x_limits=(0.0, 10.0),
                radius_base=0.6,
                radius_tip=0.4,
                curvature_limit=0.3,
                tendon_length_limits=(5.0, 15.0),
                tendon_rest_length=10.0,
                tendon_routing_radius=0.5,
                max_tendon_delta_per_step=0.02,
                render_points_per_segment=128,
            ),
            CircleObstacle(center=np.array([24.0, 0.0], dtype=float), radius=3.0),
            0.0,
            None,
            10.0,
            np.zeros((2, 2), dtype=float),
            1,
        ),
        (
            "multi_point_wrapping_block",
            ArmModel2DConfig(
                segment_lengths=(10.0, 10.0),
                base_x_limits=(0.0, 10.0),
                radius_base=0.6,
                radius_tip=0.4,
                curvature_limit=0.3,
                tendon_length_limits=(5.0, 15.0),
                tendon_rest_length=10.0,
                tendon_routing_radius=0.5,
                max_tendon_delta_per_step=0.02,
                render_points_per_segment=128,
            ),
            CircleObstacle(center=np.array([12.5, 4.0], dtype=float), radius=4.0),
            0.0,
            None,
            0.0,
            np.array([[-5.0, 5.0], [-5.0, 5.0]], dtype=float),
            8,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_severe_penetration_rolls_back_to_safe_contact_state(
    name: str,
    config: ArmModel2DConfig,
    obstacle: CircleObstacle,
    base_x: float,
    kappas: np.ndarray | None,
    base_delta: float,
    tendon_deltas: np.ndarray,
    min_boundary_samples: int,
) -> None:
    model = ArmModel2D(
        config=config,
        obstacles=[obstacle],
        base_x=base_x,
        kappas=None if kappas is None else np.asarray(kappas, dtype=float),
    )

    model.apply_action(base_delta=base_delta, tendon_length_deltas=tendon_deltas)

    assert model.action_blocked is True, f"{name} should trigger strict rollback blocking."
    assert model.contact_backend_details.get("action_blocked") is True

    _assert_polyline_stays_outside_obstacle(model, obstacle)

    distances = np.linalg.norm(model.observed_polyline() - obstacle.center[None, :], axis=1)
    boundary_touch_count = int(np.sum(distances <= obstacle.radius + 0.1))
    assert boundary_touch_count >= min_boundary_samples, (
        f"{name} expected at least {min_boundary_samples} near-boundary samples, "
        f"got {boundary_touch_count}."
    )

