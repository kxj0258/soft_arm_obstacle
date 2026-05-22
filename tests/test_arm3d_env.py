from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tasks.arm3d import (
    ArmModel3D,
    ArmModel3DConfig,
    Scenario3D,
    SoftArmObstacle3DEnv,
    SoftArmSim3DConfig,
    get_preset_scenario,
    list_preset_scenarios,
    register_soft_arm_3d_env,
)
from tdcr.datasets.schema import collate_samples, sample_from_shape
from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D
from tdcr.solvers import CosseratStaticSolver, PCCAContact3DSolver


ROOT = Path(__file__).resolve().parents[1]
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class Arm3DTests(unittest.TestCase):
    def test_zero_tension_shape_points_along_positive_z(self) -> None:
        params = TDCRRobotParams(segment_lengths=(0.08, 0.08, 0.08), samples_per_segment=8)
        model = ArmModel3D(
            config=ArmModel3DConfig(robot_params=params, solver_backend="pcc3d"),
            obstacles=[],
            tau=np.zeros((3, 3), dtype=float),
        )
        np.testing.assert_allclose(model.tip_position, np.array([0.0, 0.0, params.total_length]), atol=1e-6)
        np.testing.assert_allclose(model.tip_rotvec, np.zeros(3), atol=1e-6)
        self.assertEqual(model.tau.shape, (3, 3))

    def test_cosserat_static_solver_reports_fallback_for_default_3_segment_robot(self) -> None:
        params = TDCRRobotParams()
        solver = CosseratStaticSolver(params)
        shape = solver.solve(tau=np.zeros((3, 3), dtype=float), obstacles=[])
        self.assertEqual(shape.diagnostics["solver"], "cosserat_static")
        self.assertEqual(shape.diagnostics["status"], "fallback_to_pcc")
        self.assertIn("2-segment, 3-tendon-per-segment", str(shape.diagnostics["reason"]))

    def test_model_accepts_cosserat_static_backend_name(self) -> None:
        params = TDCRRobotParams()
        model = ArmModel3D(
            config=ArmModel3DConfig(robot_params=params, solver_backend="cosserat_static"),
            obstacles=[],
            tau=np.zeros((3, 3), dtype=float),
        )
        self.assertEqual(model.contact_backend_used, "cosserat_static")
        self.assertEqual(model.contact_backend_details["status"], "fallback_to_pcc")

    def test_env_reset_step_contract(self) -> None:
        env = SoftArmObstacle3DEnv(
            render_mode=None,
            max_steps=3,
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
            sim_config=SoftArmSim3DConfig(obstacle_count=1),
        )
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (9,))
        self.assertIn("tau", info)
        self.assertIn("orientation_error_to_target", info)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn("contact_summary", info)
        self.assertIn("evaluation_metrics", info)
        env.close()

    def test_sim_config_exposes_valid_decimation(self) -> None:
        config = SoftArmSim3DConfig(sim_dt=0.001, control_dt=0.02)
        self.assertEqual(config.decimation, 20)

    def test_decimation_one_matches_reward_decomposition(self) -> None:
        env = SoftArmObstacle3DEnv(
            render_mode=None,
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
            sim_config=SoftArmSim3DConfig(obstacle_count=0, sim_dt=0.02, control_dt=0.02),
        )
        env.reset(seed=0, options={"scenario_name": "reach_easy_3d"})
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        reward_terms = info["reward_terms"]
        recomposed_reward = (
            float(reward_terms["process_total"])
            + float(reward_terms["terminal_total"])
        )
        self.assertAlmostEqual(reward, recomposed_reward, places=7)
        self.assertEqual(int(info["contact_summary"]["micro_step_count"]), 1)
        env.close()

    def test_reset_with_explicit_scenario(self) -> None:
        env = SoftArmObstacle3DEnv(
            render_mode=None,
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
            sim_config=SoftArmSim3DConfig(obstacle_count=2),
        )
        scenario = Scenario3D(
            obstacles=[(0.02, 0.0, 0.12, 0.012)],
            target=(0.0, 0.0, 0.24),
            target_rotvec=(0.0, 0.0, 0.0),
            tau=np.zeros((3, 3), dtype=float),
        )
        observation, info = env.reset(seed=0, options={"scenario": scenario})
        np.testing.assert_allclose(info["target_position"], np.array([0.0, 0.0, 0.24]))
        np.testing.assert_allclose(info["target_rotvec"], np.zeros(3))
        self.assertEqual(observation.shape, env.observation_space.shape)
        env.close()

    def test_reset_with_preset_scenario_name(self) -> None:
        env = SoftArmObstacle3DEnv(
            render_mode=None,
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
        )
        preset_name = list_preset_scenarios()[0]
        observation, info = env.reset(seed=0, options={"scenario_name": preset_name})
        self.assertEqual(info["scenario_name"], preset_name)
        self.assertEqual(observation.shape, env.observation_space.shape)
        env.close()

    def test_rgb_render_returns_frame(self) -> None:
        env = SoftArmObstacle3DEnv(
            render_mode="rgb_array",
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
            sim_config=SoftArmSim3DConfig(obstacle_count=0),
        )
        env.reset(seed=0, options={"scenario": get_preset_scenario("reach_easy_3d")})
        frame = env.render()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertGreater(frame.shape[0], 0)
        self.assertGreater(frame.shape[1], 0)
        env.close()

    def test_registration_helper_registers_env(self) -> None:
        env_id = register_soft_arm_3d_env()
        env = __import__("gymnasium").make(
            env_id,
            render_mode=None,
            model_config=ArmModel3DConfig(solver_backend="pcc3d"),
            sim_config=SoftArmSim3DConfig(obstacle_count=0),
        )
        observation, info = env.reset(seed=0, options={"scenario_name": "reach_easy_3d"})
        self.assertEqual(observation.shape, env.observation_space.shape)
        env.close()

    def test_dataset_schema_collates_shape_samples(self) -> None:
        params = TDCRRobotParams(samples_per_segment=4)
        model = ArmModel3D(
            config=ArmModel3DConfig(robot_params=params, solver_backend="pcc3d"),
            obstacles=[Obstacle3D(center=np.array([0.02, 0.0, 0.10]), radii=0.012)],
            tau=np.zeros((3, 3), dtype=float),
        )
        sample = sample_from_shape(model.shape, robot_params=params, obstacles=model.obstacles)
        batch = collate_samples([sample, sample])
        self.assertEqual(batch.tau.shape, (2, 3, 3))
        self.assertEqual(batch.contact_slots.shape[-1], 24)
        self.assertEqual(batch.obstacle_slots.shape[-1], 10)
        self.assertTrue(np.isfinite(batch.p).all())

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the 3D PCCA contact backend")
    def test_pcca_contact_solver_runs_with_obstacle(self) -> None:
        params = TDCRRobotParams(samples_per_segment=4)
        solver = PCCAContact3DSolver(params, max_iterations=20)
        obstacle = Obstacle3D(center=np.array([0.0, 0.0, 0.11]), radii=0.018)
        shape = solver.solve(tau=np.zeros((3, 3), dtype=float), obstacles=[obstacle])
        self.assertEqual(shape.p.shape[1], 3)
        self.assertIn(shape.contact_summary.contact_state, {"free", "near", "collision"})
        self.assertTrue(np.isfinite(shape.p).all())

    def test_main_cli_runs_arm3d(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--task",
                "arm3d",
                "--solver-backend",
                "pcc3d",
                "--render-mode",
                "none",
                "--episodes",
                "1",
                "--steps",
                "3",
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
        self.assertIn("task=arm3d", output)

    def test_train_cli_smoke_runs_arm3d_model_based(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            log_dir = Path(tmpdir) / "logs"
            result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--task",
                    "arm3d",
                    "--solver-backend",
                    "pcc3d",
                    "--timesteps",
                    "3",
                    "--eval-freq",
                    "0",
                    "--n-envs",
                    "1",
                    "--rollout-steps",
                    "16",
                    "--max-episode-steps",
                    "8",
                    "--planning-horizon",
                    "2",
                    "--num-candidates",
                    "4",
                    "--num-elites",
                    "2",
                    "--num-iterations",
                    "1",
                    "--scenario-name",
                    "reach_easy_3d",
                    "--model-dir",
                    str(model_dir),
                    "--log-dir",
                    str(log_dir),
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=output)
            self.assertIn("Training finished.", output)
            self.assertTrue((model_dir / "final_model.mpc.json").exists(), msg=output)


if __name__ == "__main__":
    unittest.main()
