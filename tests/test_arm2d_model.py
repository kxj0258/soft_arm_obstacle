from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tasks.arm2d import (
    ArmModel2D,
    ArmModel2DConfig,
    CircleObstacle,
    ContactResponse2DConfig,
    get_preset_scenario,
)

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class ArmModel2DTests(unittest.TestCase):
    def test_straight_forward_kinematics_matches_total_length(self) -> None:
        config = ArmModel2DConfig(segment_lengths=(10.0, 15.0))
        model = ArmModel2D(config=config, obstacles=[])
        np.testing.assert_allclose(model.tip_position, np.array([25.0, 0.0]), atol=1e-6)
        self.assertAlmostEqual(model.tip_angle, 0.0, places=6)

    def test_collision_response_inserts_segments_without_nan(self) -> None:
        config = ArmModel2DConfig(
            segment_lengths=(10.0, 15.0),
            arm_radius=1.0,
            contact_response=ContactResponse2DConfig(contact_release_clearance=0.5),
        )
        model = ArmModel2D(config=config, obstacles=[CircleObstacle(center=np.array([13.0, 1.0]), radius=3.2)])
        self.assertGreater(len(model.observed_segments), len(model.free_segments))
        self.assertGreater(len(model.collision_events), 0)
        self.assertTrue(model.contact_summary.active)
        self.assertEqual(model.contact_summary.collision_count, len(model.collision_events))
        self.assertEqual(model.contact_summary.contact_state, "collision")
        self.assertEqual(model.contact_summary.segment_clearances.shape, (len(model.observed_segments),))
        self.assertEqual(model.contact_summary.segment_contact_mask.shape, (len(model.observed_segments),))
        for segment in model.observed_segments:
            self.assertTrue(np.isfinite(segment.start_position).all())
            self.assertTrue(np.isfinite(segment.end_position).all())

    def test_action_clips_base_and_curvature(self) -> None:
        config = ArmModel2DConfig(
            base_x_limits=(0.0, 5.0),
            curvature_limit=0.1,
            tendon_length_limits=(8.0, 12.0),
            tendon_routing_radius=0.5,
        )
        model = ArmModel2D(config=config, obstacles=[])
        model.apply_action(
            base_delta=10.0,
            tendon_length_deltas=np.array([[-4.0, 4.0], [4.0, -4.0]], dtype=float),
        )
        self.assertLessEqual(model.base_x, 5.0)
        self.assertTrue(np.all(model.free_tendon_lengths >= 8.0 - 1e-8))
        self.assertTrue(np.all(model.free_tendon_lengths <= 12.0 + 1e-8))
        self.assertTrue(np.all(np.abs(model.free_kappas) <= 0.1 + 1e-8))

    def test_tip_state_is_float32_vector(self) -> None:
        model = ArmModel2D(config=ArmModel2DConfig(), obstacles=[])
        tip_state = model.tip_state()
        self.assertEqual(tip_state.shape, (3,))
        self.assertEqual(tip_state.dtype, np.float32)

    def test_tendon_length_difference_maps_to_expected_curvature(self) -> None:
        config = ArmModel2DConfig(
            segment_lengths=(10.0, 10.0),
            curvature_limit=0.25,
            tendon_rest_length=10.0,
            tendon_routing_radius=0.5,
        )
        tendon_lengths = np.array([[9.0, 11.0], [10.0, 10.0]], dtype=float)
        model = ArmModel2D(config=config, obstacles=[], tendon_lengths=tendon_lengths)
        np.testing.assert_allclose(model.free_kappas, np.array([0.2, 0.0]), atol=1e-6)
        np.testing.assert_allclose(model.free_tendon_lengths, tendon_lengths, atol=1e-6)

    def test_contact_summary_tracks_nearest_obstacle_without_collision(self) -> None:
        model = ArmModel2D(
            config=ArmModel2DConfig(segment_lengths=(8.0, 8.0), arm_radius=1.0),
            obstacles=[CircleObstacle(center=np.array([14.0, 5.0]), radius=2.0)],
        )
        self.assertFalse(model.contact_summary.active)
        self.assertGreater(model.contact_summary.min_clearance, 0.0)
        self.assertGreaterEqual(model.contact_summary.nearest_obstacle_index, 0)
        self.assertTrue(np.isfinite(model.contact_summary.nearest_arm_point).all())
        self.assertEqual(model.contact_summary.contact_state, "free")

    def test_backbone_points_return_fixed_shape(self) -> None:
        model = ArmModel2D(config=ArmModel2DConfig(), obstacles=[])
        points = model.backbone_points(4)
        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.isfinite(points).all())

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the optimization contact backend")
    def test_optimization_contact_backend_preserves_control_segment_means(self) -> None:
        config = ArmModel2DConfig(
            segment_lengths=(9.0, 12.0),
            curvature_limit=0.12,
            contact_response=ContactResponse2DConfig(
                backend="optimization",
                optimization_subsegments_per_segment=4,
                fallback_to_heuristic=False,
            ),
        )
        free_kappas = np.array([0.06, -0.04], dtype=float)
        model = ArmModel2D(config=config, obstacles=[], kappas=free_kappas)
        self.assertEqual(model.contact_backend_used, "optimization")
        self.assertEqual(len(model.observed_segments), 8)
        self.assertFalse(model.contact_summary.active)
        observed_kappas = np.array([segment.kappa for segment in model.observed_segments], dtype=float)
        np.testing.assert_allclose(np.mean(observed_kappas[:4]), free_kappas[0], atol=1e-5)
        np.testing.assert_allclose(np.mean(observed_kappas[4:]), free_kappas[1], atol=1e-5)
        self.assertEqual(model.contact_summary.segment_clearances.shape, (2,))

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the optimization contact backend")
    def test_optimization_contact_backend_handles_contact_preset_without_large_penetration(self) -> None:
        scenario = get_preset_scenario("contact_left")
        config = ArmModel2DConfig(
            contact_response=ContactResponse2DConfig(
                backend="optimization",
                optimization_subsegments_per_segment=6,
                fallback_to_heuristic=False,
            ),
        )
        model = ArmModel2D(
            config=config,
            obstacles=scenario.obstacles,
            base_x=scenario.base_x,
            kappas=scenario.kappas,
        )
        self.assertEqual(model.contact_backend_used, "optimization")
        self.assertGreaterEqual(model.contact_summary.min_clearance, -1e-4)
        self.assertIn(model.contact_summary.contact_state, {"constrained", "free", "latched"})


if __name__ == "__main__":
    unittest.main()
