from __future__ import annotations

import unittest

import numpy as np

from tasks.arm2d import ArmModel2D, ArmModel2DConfig, CircleObstacle


class ArmModel2DTests(unittest.TestCase):
    def test_straight_forward_kinematics_matches_total_length(self) -> None:
        config = ArmModel2DConfig(segment_lengths=(10.0, 15.0))
        model = ArmModel2D(config=config, obstacles=[])
        np.testing.assert_allclose(model.tip_position, np.array([25.0, 0.0]), atol=1e-6)
        self.assertAlmostEqual(model.tip_angle, 0.0, places=6)

    def test_collision_response_inserts_segments_without_nan(self) -> None:
        config = ArmModel2DConfig(segment_lengths=(10.0, 15.0), arm_radius=1.0)
        model = ArmModel2D(config=config, obstacles=[CircleObstacle(center=np.array([13.0, 1.0]), radius=3.2)])
        self.assertGreater(len(model.observed_segments), len(model.free_segments))
        self.assertGreater(len(model.collision_events), 0)
        for segment in model.observed_segments:
            self.assertTrue(np.isfinite(segment.start_position).all())
            self.assertTrue(np.isfinite(segment.end_position).all())

    def test_action_clips_base_and_curvature(self) -> None:
        config = ArmModel2DConfig(base_x_limits=(0.0, 5.0), curvature_limit=0.1)
        model = ArmModel2D(config=config, obstacles=[])
        model.apply_action(base_delta=10.0, kappa_delta=np.array([1.0, -1.0]))
        self.assertLessEqual(model.base_x, 5.0)
        self.assertTrue(np.all(np.abs(model.free_kappas) <= 0.1 + 1e-8))


if __name__ == "__main__":
    unittest.main()
