from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tasks.arm2d import (
    ModelBasedCEMConfig,
    ModelBasedCEMController,
    Observation2DConfig,
    Render2DConfig,
    Scenario2D,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
    SoftArmSim2DConfig,
    TendonControl2DConfig,
    list_preset_scenarios,
    register_soft_arm_2d_env,
)


ROOT = Path(__file__).resolve().parents[1]


class SoftArmObstacle2DEnvTests(unittest.TestCase):
    def test_reset_is_seeded(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None)
        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)
        np.testing.assert_allclose(obs_a, obs_b)
        np.testing.assert_allclose(info_a["target_position"], info_b["target_position"])
        self.assertAlmostEqual(info_a["target_angle"], info_b["target_angle"], places=6)
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
        self.assertIn("reward_terms", info)
        self.assertIn("contact_summary", info)
        self.assertIn("evaluation_metrics", info)
        self.assertIn("contact_backend", info)
        self.assertIn("contact_backend_details", info)
        self.assertIn("tendon_lengths", info)
        self.assertIn("path_length", info["evaluation_metrics"])
        env.close()

    def test_sim_config_exposes_valid_decimation(self) -> None:
        config = SoftArmSim2DConfig(sim_dt=0.001, control_dt=0.02)
        self.assertEqual(config.decimation, 20)

    def test_decimation_one_matches_legacy_reward_decomposition(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            sim_config=SoftArmSim2DConfig(obstacle_count=0, sim_dt=0.02, control_dt=0.02),
        )
        env.reset(seed=0, options={"scenario_name": "reach_easy"})
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

    def test_contact_summary_reports_occurrence_rate_with_decimation(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            sim_config=SoftArmSim2DConfig(sim_dt=0.001, control_dt=0.01),
        )
        env.reset(seed=0, options={"scenario_name": "contact_left"})
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        self.assertIn("contact_occurrence_rate", info["contact_summary"])
        self.assertGreaterEqual(float(info["contact_summary"]["contact_occurrence_rate"]), 0.0)
        self.assertLessEqual(float(info["contact_summary"]["contact_occurrence_rate"]), 1.0)
        self.assertEqual(int(info["contact_summary"]["micro_step_count"]), 10)
        env.close()

    def test_custom_env_config_changes_action_and_observation_shapes(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            env_config=SoftArmObstacle2DEnvConfig(
                control=TendonControl2DConfig(base_delta_scale=0.5, tendon_delta_scale=0.25),
                observation=Observation2DConfig(include_collision_count=False),
                render=Render2DConfig(xlim=(-5.0, 30.0), ylim=(-10.0, 10.0)),
            ),
            sim_config=SoftArmSim2DConfig(obstacle_count=2),
        )
        observation, info = env.reset(seed=0)
        self.assertEqual(env.action_space.shape, (5,))
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_labels[0], "base_motor")
        env.close()

    def test_reset_with_explicit_scenario(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, sim_config=SoftArmSim2DConfig(obstacle_count=3))
        scenario = Scenario2D(
            obstacles=[(12.0, 4.0, 2.5)],
            target=(20.0, 0.0),
            target_angle=0.25,
            base_x=1.0,
            kappas=np.array([0.05, -0.03]),
        )
        observation, info = env.reset(seed=0, options={"scenario": scenario})
        self.assertAlmostEqual(info["base_x"], 1.0, places=6)
        np.testing.assert_allclose(info["target_position"], np.array([20.0, 0.0]))
        self.assertAlmostEqual(info["target_angle"], 0.25, places=6)
        self.assertEqual(observation.shape, env.observation_space.shape)
        env.close()

    def test_reset_with_preset_scenario_name(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None)
        preset_name = list_preset_scenarios()[0]
        observation, info = env.reset(seed=0, options={"scenario_name": preset_name})
        self.assertEqual(info["scenario_name"], preset_name)
        self.assertEqual(observation.shape, env.observation_space.shape)
        env.close()

    def test_relative_observation_mode_and_reward_terms(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            env_config=SoftArmObstacle2DEnvConfig(
                observation=Observation2DConfig(mode="relative", backbone_points=3),
            ),
        )
        env.reset(seed=0)
        observation, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertIn("contact_progress_bonus", info["reward_terms"])
        self.assertIn("idle_contact_penalty", info["reward_terms"])
        self.assertIn("overlap_penalty", info["reward_terms"])
        self.assertIn("angle_progress", info["reward_terms"])
        self.assertIn("angle_error", info["reward_terms"])
        self.assertIn("evaluation_metrics", info)
        self.assertIn("contact_ratio", info["evaluation_metrics"])
        env.close()

    def test_pose_goal_requires_orientation_alignment(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, sim_config=SoftArmSim2DConfig(obstacle_count=0))
        scenario = Scenario2D(
            obstacles=[],
            target=(20.0, 0.0),
            target_angle=np.pi / 2.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        )
        env.reset(seed=0, options={"scenario": scenario})
        action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(info["success"])
        self.assertGreater(abs(info["angle_error_to_target"]), info["target_angle_tolerance"])
        env.close()

    def test_pose_goal_succeeds_when_position_and_orientation_match(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, sim_config=SoftArmSim2DConfig(obstacle_count=0))
        scenario = Scenario2D(
            obstacles=[],
            target=(20.0, 0.0),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        )
        env.reset(seed=0, options={"scenario": scenario})
        action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertLessEqual(abs(info["angle_error_to_target"]), info["target_angle_tolerance"])
        env.close()

    def test_observation_with_backbone_and_segment_contact_features(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            env_config=SoftArmObstacle2DEnvConfig(
                observation=Observation2DConfig(
                    mode="hybrid",
                    backbone_points=5,
                    include_backbone_points=True,
                    include_segment_contact_features=True,
                ),
            ),
        )
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(observation).all())
        env.close()

    def test_legacy_observation_mode_matches_expected_shape(self) -> None:
        env = SoftArmObstacle2DEnv(
            render_mode=None,
            env_config=SoftArmObstacle2DEnvConfig(
                observation=Observation2DConfig(
                    mode="legacy_v1",
                    include_contact_features=False,
                    include_backbone_points=False,
                    include_segment_contact_features=False,
                    normalize=False,
                    low=-128.0,
                    high=128.0,
                ),
            ),
        )
        observation, info = env.reset(seed=0)
        expected_dim = env.observation_config.observation_dim(
            num_kappas=env.num_kappas,
            max_obstacles=env.max_obstacles,
        )
        self.assertEqual(observation.shape, (expected_dim,))
        self.assertEqual(observation.shape, env.observation_space.shape)
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

    def test_registration_helper_registers_env(self) -> None:
        env_id = register_soft_arm_2d_env()
        env = __import__("gymnasium").make(env_id, render_mode=None)
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, env.observation_space.shape)
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

    def test_train_cli_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("--timesteps", output)
        self.assertIn("--algorithm", output)

    def test_model_based_controller_returns_valid_action(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, sim_config=SoftArmSim2DConfig(obstacle_count=0))
        env.reset(seed=0)
        controller = ModelBasedCEMController(
            sim_template=env.sim,
            control_config=env.control_config,
            reward_config=env.reward_config,
            config=ModelBasedCEMConfig(planning_horizon=3, num_candidates=8, num_elites=2, num_iterations=1),
            seed=0,
        )
        action = controller.act(env.sim)
        self.assertEqual(action.shape, env.action_space.shape)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.all(action >= env.action_space.low))
        self.assertTrue(np.all(action <= env.action_space.high))
        self.assertIn("optimization_event_count", controller.last_plan_info)
        env.close()

    def test_train_cli_smoke_runs_without_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            log_dir = Path(tmpdir) / "logs"
            result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--timesteps",
                    "16",
                    "--eval-freq",
                    "0",
                    "--n-envs",
                    "1",
                    "--rollout-steps",
                    "16",
                    "--max-episode-steps",
                    "64",
                    "--observation-mode",
                    "relative",
                    "--backbone-points",
                    "5",
                    "--planning-horizon",
                    "3",
                    "--num-candidates",
                    "8",
                    "--num-elites",
                    "2",
                    "--num-iterations",
                    "1",
                    "--scenario-name",
                    "reach_easy",
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

    def test_train_cli_subproc_smoke_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            log_dir = Path(tmpdir) / "logs"
            result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--timesteps",
                    "16",
                    "--eval-freq",
                    "0",
                    "--n-envs",
                    "2",
                    "--rollout-steps",
                    "16",
                    "--max-episode-steps",
                    "32",
                    "--planning-horizon",
                    "3",
                    "--num-candidates",
                    "8",
                    "--num-elites",
                    "2",
                    "--num-iterations",
                    "1",
                    "--scenario-name",
                    "reach_easy",
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

    def test_play_cli_smoke_runs_with_saved_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            log_dir = Path(tmpdir) / "logs"
            train_result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--timesteps",
                    "16",
                    "--eval-freq",
                    "0",
                    "--n-envs",
                    "1",
                    "--rollout-steps",
                    "16",
                    "--max-episode-steps",
                    "32",
                    "--planning-horizon",
                    "3",
                    "--num-candidates",
                    "8",
                    "--num-elites",
                    "2",
                    "--num-iterations",
                    "1",
                    "--scenario-name",
                    "reach_easy",
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
            train_output = train_result.stdout + train_result.stderr
            self.assertEqual(train_result.returncode, 0, msg=train_output)

            play_result = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--play",
                    "--model-path",
                    str(model_dir / "final_model"),
                    "--play-episodes",
                    "1",
                    "--play-render-mode",
                    "none",
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            play_output = play_result.stdout + play_result.stderr
            self.assertEqual(play_result.returncode, 0, msg=play_output)
            self.assertIn("Episode 1 finished.", play_output)

    def test_zero_action_rollout_performance_smoke(self) -> None:
        env = SoftArmObstacle2DEnv(render_mode=None, max_steps=64)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.reset(seed=0)
        start = time.perf_counter()
        for _ in range(80):
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset(seed=0)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, msg=f"80 zero-action steps took {elapsed:.3f}s")
        env.close()


if __name__ == "__main__":
    unittest.main()
