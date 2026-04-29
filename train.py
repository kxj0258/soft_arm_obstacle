import os
import json
import argparse
import gymnasium as gym
from tasks.arm2d import (
    ArmModel2DConfig,
    ContactResponse2DConfig,
    Observation2DConfig,
    SOFT_ARM_2D_ENV_ID,
    SoftArmObstacle2DEnvConfig,
    list_preset_scenarios,
    register_soft_arm_2d_env,
)


PRESET_SCENARIO_CHOICES = list_preset_scenarios()


def _build_observation_config(
    *,
    observation_mode: str,
    backbone_points: int,
    include_backbone_points: bool,
    include_segment_contact_features: bool,
) -> Observation2DConfig:
    return Observation2DConfig(
        mode=observation_mode,
        backbone_points=max(int(backbone_points), 0),
        include_backbone_points=bool(include_backbone_points),
        include_segment_contact_features=bool(include_segment_contact_features),
    )


def _run_metadata(args) -> dict:
    return {
        "max_episode_steps": int(args.max_episode_steps),
        "scenario_name": args.scenario_name,
        "contact_backend": args.contact_backend,
        "observation": {
            "mode": args.observation_mode,
            "backbone_points": max(int(args.backbone_points), 0),
            "include_backbone_points": not args.disable_backbone_points,
            "include_segment_contact_features": not args.disable_segment_contact_features,
        },
    }


def _save_run_metadata(model_dir: str, args) -> None:
    path = os.path.join(model_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_run_metadata(args), handle, indent=2, sort_keys=True)


def _load_run_metadata(model_path: str) -> dict | None:
    candidates = [
        model_path + ".env.json",
        os.path.join(os.path.dirname(model_path), "run_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _infer_observation_config_from_model(model) -> Observation2DConfig:
    shape = getattr(model.observation_space, "shape", None)
    if not shape or len(shape) != 1:
        raise ValueError(f"Unsupported observation space for play mode: {shape!r}")

    obs_dim = int(shape[0])
    if obs_dim == 18:
        return Observation2DConfig(
            mode="legacy_v1",
            include_contact_features=False,
            include_backbone_points=False,
            include_segment_contact_features=False,
            normalize=False,
            low=-128.0,
            high=128.0,
        )

    default_config = Observation2DConfig()
    if obs_dim == default_config.observation_dim(num_kappas=2, max_obstacles=3):
        return default_config

    raise ValueError(
        "Unable to infer a compatible observation config for the loaded model. "
        f"Model expects observation dim {obs_dim}. "
        "Provide a matching run_config.json next to the model or retrain with the current code."
    )


def _resolve_play_settings(args, model) -> tuple[Observation2DConfig, int, str | None, str]:
    metadata = _load_run_metadata(args.model_path)
    if metadata is not None and "observation" in metadata:
        observation_config = Observation2DConfig(**metadata["observation"])
        default_steps = build_parser().get_default("max_episode_steps")
        max_episode_steps = (
            int(metadata.get("max_episode_steps", args.max_episode_steps))
            if args.max_episode_steps == default_steps
            else int(args.max_episode_steps)
        )
        scenario_name = args.scenario_name if args.scenario_name is not None else metadata.get("scenario_name")
        return observation_config, max_episode_steps, scenario_name, metadata.get("contact_backend", "heuristic")

    return _infer_observation_config_from_model(model), int(args.max_episode_steps), args.scenario_name, args.contact_backend

def build_parser():
    parser = argparse.ArgumentParser(description="Train Soft Arm with Tendon-driven RL Agent")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate the agent every X steps")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per evaluation run (0 disables evaluation callback setup)")
    parser.add_argument("--play", action="store_true", help="Play the best trained model instead of training")
    parser.add_argument("--model-path", type=str, default="models/best_model", help="Path to saved model")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard and evaluations")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel training environments")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to Stable-Baselines3")
    parser.add_argument("--rollout-steps", type=int, default=256, help="PPO rollout horizon per environment")
    parser.add_argument("--play-episodes", type=int, default=3, help="Number of episodes to play in --play mode")
    parser.add_argument(
        "--play-render-mode",
        type=str,
        choices=("human", "rgb_array", "none"),
        default="human",
        help="Render mode used during --play",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        choices=("absolute", "relative", "hybrid"),
        default="hybrid",
        help="Observation feature layout used by every environment instance",
    )
    parser.add_argument("--backbone-points", type=int, default=4, help="Number of fixed backbone keypoints exposed to the policy")
    parser.add_argument("--disable-backbone-points", action="store_true", help="Disable backbone keypoint features in the observation")
    parser.add_argument(
        "--disable-segment-contact-features",
        action="store_true",
        help="Disable per-segment clearance/contact mask features in the observation",
    )
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Environment horizon used during training/play")
    parser.add_argument(
        "--contact-backend",
        type=str,
        choices=("heuristic", "optimization"),
        default="heuristic",
        help="Contact resolution backend used by the arm model",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        choices=PRESET_SCENARIO_CHOICES,
        default=None,
        help="Optional preset scenario name for fully reproducible training/evaluation runs",
    )
    return parser

def _import_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    return PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv

class ScenarioResetWrapper(gym.Wrapper):
    def __init__(self, env, reset_options: dict | None = None):
        super().__init__(env)
        self._reset_options = dict(reset_options or {})

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        merged_options = dict(self._reset_options)
        if options:
            merged_options.update(options)
        return self.env.reset(seed=seed, options=merged_options or None)


def make_env(
    *,
    render_mode=None,
    seed: int | None = None,
    max_episode_steps: int = 200,
    observation_mode: str = "hybrid",
    backbone_points: int = 4,
    include_backbone_points: bool = True,
    include_segment_contact_features: bool = True,
    contact_backend: str = "heuristic",
    scenario_name: str | None = None,
):
    env_config = SoftArmObstacle2DEnvConfig(
        observation=_build_observation_config(
            observation_mode=observation_mode,
            backbone_points=backbone_points,
            include_backbone_points=include_backbone_points,
            include_segment_contact_features=include_segment_contact_features,
        ),
    )
    model_config = ArmModel2DConfig(
        contact_response=ContactResponse2DConfig(backend=contact_backend),
    )
    reset_options = {"scenario_name": scenario_name} if scenario_name is not None else None

    def _factory():
        PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
        env_id = register_soft_arm_2d_env()
        env = gym.make(
            env_id,
            render_mode=render_mode,
            max_steps=max_episode_steps,
            env_config=env_config,
            model_config=model_config,
        )
        env = ScenarioResetWrapper(env, reset_options=reset_options)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _factory

def train(args):
    PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
    register_soft_arm_2d_env()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _save_run_metadata(args.model_dir, args)

    num_envs = max(1, int(args.n_envs))
    env_fns = [
        make_env(
            render_mode=None,
            seed=args.seed + index,
            max_episode_steps=args.max_episode_steps,
            observation_mode=args.observation_mode,
            backbone_points=args.backbone_points,
            include_backbone_points=not args.disable_backbone_points,
            include_segment_contact_features=not args.disable_segment_contact_features,
            contact_backend=args.contact_backend,
            scenario_name=args.scenario_name,
        )
        for index in range(num_envs)
    ]
    train_env = DummyVecEnv(env_fns) if num_envs == 1 else SubprocVecEnv(env_fns)
    eval_env = None
    rollout_steps = max(16, min(int(args.rollout_steps), max(16, args.timesteps // num_envs)))
    batch_size = min(64, rollout_steps * num_envs)

    eval_callback = None
    if args.eval_freq > 0 and args.eval_episodes > 0:
        eval_env = DummyVecEnv(
            [
                make_env(
                    render_mode=None,
                    seed=args.seed + num_envs,
                    max_episode_steps=args.max_episode_steps,
                    observation_mode=args.observation_mode,
                    backbone_points=args.backbone_points,
                    include_backbone_points=not args.disable_backbone_points,
                    include_segment_contact_features=not args.disable_segment_contact_features,
                    contact_backend=args.contact_backend,
                    scenario_name=args.scenario_name,
                )
            ]
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=args.model_dir,
            log_path=args.log_dir,
            eval_freq=max(args.eval_freq // num_envs, 1),
            deterministic=True,
            render=False,
            n_eval_episodes=max(int(args.eval_episodes), 1),
        )

    print("Initializing PPO model for Tendon-Driven Soft Arm...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=os.path.join(args.log_dir, "ppo_softarm"),
        device=args.device,
        seed=args.seed,
        n_steps=rollout_steps,
        batch_size=batch_size,
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    
    model.save(os.path.join(args.model_dir, "final_model"))
    print("Training finished.")
    train_env.close()
    if eval_env is not None:
        eval_env.close()

def play(args):
    PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
    register_soft_arm_2d_env()
    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model file not found at {args.model_path}.zip. Please train first.")
    
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path, device=args.device)
    observation_config, max_episode_steps, scenario_name, contact_backend = _resolve_play_settings(args, model)
    render_mode = None if args.play_render_mode == "none" else args.play_render_mode
    env = gym.make(
        SOFT_ARM_2D_ENV_ID,
        render_mode=render_mode,
        max_steps=max_episode_steps,
        env_config=SoftArmObstacle2DEnvConfig(
            observation=observation_config,
        ),
        model_config=ArmModel2DConfig(
            contact_response=ContactResponse2DConfig(backend=contact_backend),
        ),
    )
    env = ScenarioResetWrapper(
        env,
        reset_options={"scenario_name": scenario_name} if scenario_name is not None else None,
    )
    try:
        for episode in range(args.play_episodes):
            obs, info = env.reset(seed=args.seed + episode)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if render_mode == "rgb_array":
                    env.render()
                if terminated or truncated:
                    print(
                        f"Episode {episode + 1} finished. "
                        f"Target Reached: {info['success']} Distance: {info['distance_to_target']:.3f}"
                    )
                    break
    finally:
        env.close()

if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.play:
        play(args)
    else:
        train(args)
