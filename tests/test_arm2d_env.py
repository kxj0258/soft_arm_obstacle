from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tasks.arm2d import SoftArmObstacle2DEnv


ROOT = Path(__file__).resolve().parents[1]


class SoftArmObstacle2DEnvTests(unittest.TestCase):
    def test_reset_is_seeded(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None)
        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)
        np.testing.assert_allclose(obs_a, obs_b)
        np.testing.assert_allclose(info_a["target_position"], info_b["target_position"])
        env.close()

    def test_step_contract_and_truncation(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, max_steps=3)
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        env.close()

    def test_rgb_render_returns_frame(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertGreater(frame.shape[0], 0)
        self.assertGreater(frame.shape[1], 0)
        env.close()

    def test_main_cli_runs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--render-mode",
                "none",
                "--episodes",
                "1",
                "--steps",
                "5",
                "--seed",
                "7",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("episode=1", output)


if __name__ == "__main__":
    unittest.main()
