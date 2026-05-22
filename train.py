import os
import json
import argparse
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym

from tasks.arm2d import (
    ArmModel2DConfig,
    ContactResponse2DConfig,
    ModelBasedCEMConfig,
    ModelBasedCEMController,
    Observation2DConfig,
    SOFT_ARM_2D_ENV_ID,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
    SoftArmSim2DConfig,
    list_preset_scenarios,
    register_soft_arm_2d_env,
)
from tasks.arm3d import (
    ArmModel3DConfig,
    ModelBasedCEM3DConfig,
    ModelBasedCEM3DController,
    Observation3DConfig,
    SOFT_ARM_3D_ENV_ID,
    SoftArmObstacle3DEnv,
    SoftArmObstacle3DEnvConfig,
    SoftArmSim3DConfig,
    list_preset_scenarios as list_preset_scenarios_3d,
    register_soft_arm_3d_env,
)


PRESET_SCENARIO_CHOICES = sorted(set(list_preset_scenarios()) | set(list_preset_scenarios_3d()))
PPO_MODEL_SUFFIX = ".zip"
MODEL_BASED_MODEL_SUFFIX = ".mpc.json"


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


def _build_observation_config_3d(
    *,
    backbone_points: int,
    include_backbone_points: bool,
    include_segment_contact_features: bool,
) -> Observation3DConfig:
    return Observation3DConfig(
        backbone_points=max(int(backbone_points), 0),
        include_backbone_points=bool(include_backbone_points),
        include_segment_contact_features=bool(include_segment_contact_features),
    )


def _build_model_based_config(args) -> ModelBasedCEMConfig:
    return ModelBasedCEMConfig(
        planning_horizon=max(int(args.planning_horizon), 1),
        num_candidates=max(int(args.num_candidates), 1),
        num_elites=max(int(args.num_elites), 1),
        num_iterations=max(int(args.num_iterations), 1),
        gamma=float(args.planning_gamma),
        init_std=float(args.planning_init_std),
        min_std=float(args.planning_min_std),
        momentum=float(args.planning_momentum),
        terminal_distance_weight=float(args.terminal_distance_weight),
        terminal_angle_weight=float(args.terminal_angle_weight),
    )


def _build_model_based_config_3d(args) -> ModelBasedCEM3DConfig:
    return ModelBasedCEM3DConfig(
        planning_horizon=max(int(args.planning_horizon), 1),
        num_candidates=max(int(args.num_candidates), 1),
        num_elites=max(int(args.num_elites), 1),
        num_iterations=max(int(args.num_iterations), 1),
        gamma=float(args.planning_gamma),
        init_std=float(args.planning_init_std),
        min_std=float(args.planning_min_std),
        momentum=float(args.planning_momentum),
        terminal_distance_weight=float(args.terminal_distance_weight),
        terminal_orientation_weight=float(args.terminal_angle_weight),
    )


def _run_metadata(args) -> dict:
    return {
        "algorithm": args.algorithm,
        "task": args.task,
        "max_episode_steps": int(args.max_episode_steps),
        "scenario_name": args.scenario_name,
        "contact_backend": args.contact_backend,
        "solver_backend": args.solver_backend,
        "observation": {
            "mode": args.observation_mode,
            "backbone_points": max(int(args.backbone_points), 0),
            "include_backbone_points": not args.disable_backbone_points,
            "include_segment_contact_features": not args.disable_segment_contact_features,
        },
        "model_based_cem": asdict(_build_model_based_config(args)),
        "model_based_cem_3d": asdict(_build_model_based_config_3d(args)),
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
    current_max_obstacles = SoftArmSim2DConfig().obstacle_count
    legacy_config = Observation2DConfig(
        mode="legacy_v1",
        include_contact_features=False,
        include_backbone_points=False,
        include_segment_contact_features=False,
        normalize=False,
        low=-128.0,
        high=128.0,
    )
    default_config = Observation2DConfig()

    if obs_dim == legacy_config.observation_dim(num_kappas=2, max_obstacles=current_max_obstacles):
        return legacy_config
    if obs_dim == default_config.observation_dim(num_kappas=2, max_obstacles=current_max_obstacles):
        return default_config
    if obs_dim in {
        legacy_config.observation_dim(num_kappas=2, max_obstacles=3),
        default_config.observation_dim(num_kappas=2, max_obstacles=3),
    }:
        raise ValueError(
            "Loaded model appears to use an older 3-obstacle observation layout. "
            "Please place the matching run_config.json next to the model or replay it with the older code."
        )

    raise ValueError(
        "Unable to infer a compatible observation config for the loaded model. "
        f"Model expects observation dim {obs_dim}. "
        "Provide a matching run_config.json next to the model or retrain with the current code."
    )


def _resolve_env_settings(args, metadata: dict | None) -> tuple[Observation2DConfig, int, str | None, str]:
    if metadata is None:
        return (
            _build_observation_config(
                observation_mode=args.observation_mode,
                backbone_points=args.backbone_points,
                include_backbone_points=not args.disable_backbone_points,
                include_segment_contact_features=not args.disable_segment_contact_features,
            ),
            int(args.max_episode_steps),
            args.scenario_name,
            args.contact_backend,
        )

    observation_payload = metadata.get("observation")
    observation_config = (
        Observation2DConfig(**observation_payload)
        if observation_payload is not None
        else _build_observation_config(
            observation_mode=args.observation_mode,
            backbone_points=args.backbone_points,
            include_backbone_points=not args.disable_backbone_points,
            include_segment_contact_features=not args.disable_segment_contact_features,
        )
    )
    default_steps = build_parser().get_default("max_episode_steps")
    max_episode_steps = (
        int(metadata.get("max_episode_steps", args.max_episode_steps))
        if args.max_episode_steps == default_steps
        else int(args.max_episode_steps)
    )
    scenario_name = args.scenario_name if args.scenario_name is not None else metadata.get("scenario_name")
    contact_backend = metadata.get("contact_backend", args.contact_backend)
    return observation_config, max_episode_steps, scenario_name, contact_backend


def _resolve_task(args, metadata: dict | None) -> str:
    task = args.task if metadata is None else metadata.get("task", args.task)
    if task not in {"arm2d", "arm3d"}:
        raise ValueError(f"Unsupported task in metadata: {task!r}.")
    return str(task)


def _resolve_env_settings_3d(args, metadata: dict | None) -> tuple[Observation3DConfig, int, str | None, str]:
    if metadata is None:
        return (
            _build_observation_config_3d(
                backbone_points=args.backbone_points,
                include_backbone_points=not args.disable_backbone_points,
                include_segment_contact_features=not args.disable_segment_contact_features,
            ),
            int(args.max_episode_steps),
            args.scenario_name,
            args.solver_backend,
        )

    observation_payload = metadata.get("observation") or {}
    observation_config = Observation3DConfig(
        backbone_points=max(int(observation_payload.get("backbone_points", args.backbone_points)), 0),
        include_backbone_points=bool(
            observation_payload.get("include_backbone_points", not args.disable_backbone_points)
        ),
        include_segment_contact_features=bool(
            observation_payload.get("include_segment_contact_features", not args.disable_segment_contact_features)
        ),
    )
    default_steps = build_parser().get_default("max_episode_steps")
    max_episode_steps = (
        int(metadata.get("max_episode_steps", args.max_episode_steps))
        if args.max_episode_steps == default_steps
        else int(args.max_episode_steps)
    )
    scenario_name = args.scenario_name if args.scenario_name is not None else metadata.get("scenario_name")
    solver_backend = metadata.get("solver_backend", args.solver_backend)
    return observation_config, max_episode_steps, scenario_name, solver_backend


def _resolve_model_based_config(args, metadata: dict | None) -> ModelBasedCEMConfig:
    if metadata is not None and "model_based_cem" in metadata:
        return ModelBasedCEMConfig(**metadata["model_based_cem"])
    return _build_model_based_config(args)


def _resolve_model_based_config_3d(args, metadata: dict | None) -> ModelBasedCEM3DConfig:
    if metadata is not None and "model_based_cem_3d" in metadata:
        return ModelBasedCEM3DConfig(**metadata["model_based_cem_3d"])
    return _build_model_based_config_3d(args)


def _resolve_model_artifact(model_path: str) -> tuple[str, str]:
    direct_path = Path(model_path)
    if direct_path.exists():
        if direct_path.name.endswith(PPO_MODEL_SUFFIX):
            return "ppo", str(direct_path)
        if direct_path.name.endswith(MODEL_BASED_MODEL_SUFFIX):
            return "model_based_cem", str(direct_path)
    ppo_path = model_path + PPO_MODEL_SUFFIX
    if os.path.exists(ppo_path):
        return "ppo", ppo_path
    model_based_path = model_path + MODEL_BASED_MODEL_SUFFIX
    if os.path.exists(model_based_path):
        return "model_based_cem", model_based_path
    raise FileNotFoundError(
        f"Could not find a PPO model at {ppo_path} or a model-based controller at {model_based_path}."
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Train or play a soft-arm controller.")
    parser.add_argument("--task", choices=("arm2d", "arm3d"), default="arm2d", help="Environment family to train/play")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=("model_based_cem", "ppo"),
        default="model_based_cem",
        help="Controller family. model_based_cem uses simulator rollouts and CEM MPC; ppo keeps the legacy model-free baseline.",
    )
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total environment steps budget")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate the PPO agent every X steps")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per PPO evaluation run (0 disables evaluation callback setup)")
    parser.add_argument("--play", action="store_true", help="Play a saved controller instead of training")
    parser.add_argument("--model-path", type=str, default="models/best_model", help="Path stem to a saved PPO model or model-based controller")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard and evaluations")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of PPO training environments; ignored by model_based_cem")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to Stable-Baselines3 PPO")
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
    parser.add_argument("--backbone-points", type=int, default=4, help="Number of fixed backbone keypoints exposed to the controller")
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
        "--solver-backend",
        type=str,
        choices=("pcc3d", "cosserat_static", "pcca_contact_3d", "pinn", "hybrid", "oracle"),
        default="pcc3d",
        help="3D TDCR solver backend; ignored for arm2d",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        choices=PRESET_SCENARIO_CHOICES,
        default=None,
        help="Optional preset scenario name for fully reproducible runs",
    )
    parser.add_argument("--planning-horizon", type=int, default=6, help="CEM planning horizon used by model_based_cem")
    parser.add_argument("--num-candidates", type=int, default=48, help="Number of sampled action sequences per CEM iteration")
    parser.add_argument("--num-elites", type=int, default=8, help="Number of elite sequences retained per CEM iteration")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of CEM update iterations per control step")
    parser.add_argument("--planning-gamma", type=float, default=0.98, help="Discount factor used inside model_based_cem rollouts")
    parser.add_argument("--planning-init-std", type=float, default=0.6, help="Initial sampling std used by CEM")
    parser.add_argument("--planning-min-std", type=float, default=0.05, help="Lower bound on CEM sampling std")
    parser.add_argument("--planning-momentum", type=float, default=0.2, help="Momentum used when updating the CEM mean sequence")
    parser.add_argument("--terminal-distance-weight", type=float, default=0.35, help="Terminal distance shaping weight used by model_based_cem")
    parser.add_argument("--terminal-angle-weight", type=float, default=0.25, help="Terminal angle shaping weight used by model_based_cem")
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


def _make_reset_options(scenario_name: str | None) -> dict | None:
    return {"scenario_name": scenario_name} if scenario_name is not None else None


def _make_direct_env(
    *,
    task: str = "arm2d",
    render_mode=None,
    max_episode_steps: int,
    observation_config: Observation2DConfig | Observation3DConfig,
    contact_backend: str,
    solver_backend: str = "pcc3d",
) -> SoftArmObstacle2DEnv | SoftArmObstacle3DEnv:
    if task == "arm3d":
        env_config = SoftArmObstacle3DEnvConfig(observation=observation_config)
        model_config = ArmModel3DConfig(solver_backend=solver_backend)
        return SoftArmObstacle3DEnv(
            render_mode=render_mode,
            max_steps=max_episode_steps,
            env_config=env_config,
            model_config=model_config,
        )
    env_config = SoftArmObstacle2DEnvConfig(observation=observation_config)
    model_config = ArmModel2DConfig(contact_response=ContactResponse2DConfig(backend=contact_backend))
    return SoftArmObstacle2DEnv(
        render_mode=render_mode,
        max_steps=max_episode_steps,
        env_config=env_config,
        model_config=model_config,
    )


def make_env(
    *,
    task: str = "arm2d",
    render_mode=None,
    seed: int | None = None,
    max_episode_steps: int = 200,
    observation_mode: str = "hybrid",
    backbone_points: int = 4,
    include_backbone_points: bool = True,
    include_segment_contact_features: bool = True,
    contact_backend: str = "heuristic",
    solver_backend: str = "pcc3d",
    scenario_name: str | None = None,
):
    if task == "arm3d":
        env_config = SoftArmObstacle3DEnvConfig(
            observation=_build_observation_config_3d(
                backbone_points=backbone_points,
                include_backbone_points=include_backbone_points,
                include_segment_contact_features=include_segment_contact_features,
            ),
        )
        model_config = ArmModel3DConfig(solver_backend=solver_backend)
    else:
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
    reset_options = _make_reset_options(scenario_name)

    def _factory():
        PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
        env_id = register_soft_arm_3d_env() if task == "arm3d" else register_soft_arm_2d_env()
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


def _train_ppo(args) -> None:
    PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
    if args.task == "arm3d":
        register_soft_arm_3d_env()
    else:
        register_soft_arm_2d_env()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _save_run_metadata(args.model_dir, args)

    num_envs = max(1, int(args.n_envs))
    env_fns = [
        make_env(
            task=args.task,
            render_mode=None,
            seed=args.seed + index,
            max_episode_steps=args.max_episode_steps,
            observation_mode=args.observation_mode,
            backbone_points=args.backbone_points,
            include_backbone_points=not args.disable_backbone_points,
            include_segment_contact_features=not args.disable_segment_contact_features,
            contact_backend=args.contact_backend,
            solver_backend=args.solver_backend,
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
                    task=args.task,
                    render_mode=None,
                    seed=args.seed + num_envs,
                    max_episode_steps=args.max_episode_steps,
                    observation_mode=args.observation_mode,
                    backbone_points=args.backbone_points,
                    include_backbone_points=not args.disable_backbone_points,
                    include_segment_contact_features=not args.disable_segment_contact_features,
                    contact_backend=args.contact_backend,
                    solver_backend=args.solver_backend,
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

    print(f"Initializing PPO model for Tendon-Driven Soft Arm ({args.task})...")
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

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(args.model_dir, "final_model"))
    print("Training finished.")
    train_env.close()
    if eval_env is not None:
        eval_env.close()


def _train_model_based_cem(args) -> None:
    if args.task == "arm3d":
        _train_model_based_cem_3d(args)
        return

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _save_run_metadata(args.model_dir, args)

    if int(args.n_envs) != 1:
        print("model_based_cem uses a single simulator instance; ignoring --n-envs and running with one environment.")

    observation_config = _build_observation_config(
        observation_mode=args.observation_mode,
        backbone_points=args.backbone_points,
        include_backbone_points=not args.disable_backbone_points,
        include_segment_contact_features=not args.disable_segment_contact_features,
    )
    env = _make_direct_env(
        task="arm2d",
        render_mode=None,
        max_episode_steps=int(args.max_episode_steps),
        observation_config=observation_config,
        contact_backend=args.contact_backend,
    )
    reset_options = _make_reset_options(args.scenario_name)
    controller = ModelBasedCEMController(
        sim_template=env.sim,
        control_config=env.control_config,
        reward_config=env.reward_config,
        config=_build_model_based_config(args),
        seed=args.seed,
    )

    total_steps = 0
    episode_index = 0
    best_return = float("-inf")
    try:
        while total_steps < int(args.timesteps):
            _, info = env.reset(seed=args.seed + episode_index, options=reset_options)
            controller.reset()
            episode_return = 0.0
            while total_steps < int(args.timesteps):
                action = controller.act(env.sim)
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                total_steps += 1
                if terminated or truncated:
                    break

            if episode_return >= best_return:
                best_return = episode_return
                controller.save(os.path.join(args.model_dir, "best_model" + MODEL_BASED_MODEL_SUFFIX))
            plan_summary = controller.episode_optimization_event_summary()
            print(
                f"episode={episode_index + 1} steps={info['evaluation_metrics']['steps']} "
                f"return={episode_return:.3f} success={info['success']} "
                f"distance={info['distance_to_target']:.3f} angle_error={info['angle_error_to_target']:.3f} "
                f"backend={info['contact_backend']} planner_best={controller.last_plan_info.get('best_return', float('nan')):.3f} "
                f"planner_opt_events={int(plan_summary['event_count'])} "
                f"planner_penetrations={int(plan_summary['penetrating_shape_count'])} "
                f"planner_solver_failures={int(plan_summary['solver_failure_count'])} "
                f"planner_worst_clearance={plan_summary['worst_min_clearance']:.3e}"
            )
            episode_index += 1
    finally:
        controller.save(os.path.join(args.model_dir, "final_model" + MODEL_BASED_MODEL_SUFFIX))
        env.close()

    print("Training finished.")


def _train_model_based_cem_3d(args) -> None:
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _save_run_metadata(args.model_dir, args)

    if int(args.n_envs) != 1:
        print("3D model_based_cem uses a single simulator instance; ignoring --n-envs and running with one environment.")

    observation_config = _build_observation_config_3d(
        backbone_points=args.backbone_points,
        include_backbone_points=not args.disable_backbone_points,
        include_segment_contact_features=not args.disable_segment_contact_features,
    )
    env = _make_direct_env(
        task="arm3d",
        render_mode=None,
        max_episode_steps=int(args.max_episode_steps),
        observation_config=observation_config,
        contact_backend=args.contact_backend,
        solver_backend=args.solver_backend,
    )
    reset_options = _make_reset_options(args.scenario_name)
    controller = ModelBasedCEM3DController(
        sim_template=env.sim,
        control_config=env.control_config,
        reward_config=env.reward_config,
        config=_build_model_based_config_3d(args),
        seed=args.seed,
    )

    total_steps = 0
    episode_index = 0
    best_return = float("-inf")
    try:
        while total_steps < int(args.timesteps):
            _, info = env.reset(seed=args.seed + episode_index, options=reset_options)
            controller.reset()
            episode_return = 0.0
            while total_steps < int(args.timesteps):
                action = controller.act(env.sim)
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                total_steps += 1
                if terminated or truncated:
                    break

            if episode_return >= best_return:
                best_return = episode_return
                controller.save(os.path.join(args.model_dir, "best_model" + MODEL_BASED_MODEL_SUFFIX))
            print(
                f"episode={episode_index + 1} steps={info['evaluation_metrics']['steps']} "
                f"return={episode_return:.3f} success={info['success']} "
                f"distance={info['distance_to_target']:.4f} "
                f"orientation_error={info['orientation_error_to_target']:.3f} "
                f"backend={info['contact_backend']} "
                f"planner_best={controller.last_plan_info.get('best_return', float('nan')):.3f}"
            )
            episode_index += 1
    finally:
        controller.save(os.path.join(args.model_dir, "final_model" + MODEL_BASED_MODEL_SUFFIX))
        env.close()

    print("Training finished.")


def train(args):
    if args.algorithm == "ppo":
        _train_ppo(args)
        return
    if args.algorithm == "model_based_cem":
        _train_model_based_cem(args)
        return
    raise ValueError(f"Unsupported algorithm {args.algorithm!r}.")


def _play_ppo(args, artifact_path: str, metadata: dict | None) -> None:
    PPO, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _import_sb3()
    print(f"Loading PPO model from {artifact_path}...")
    model = PPO.load(artifact_path, device=args.device)

    task = _resolve_task(args, metadata)
    if task == "arm3d":
        observation_config, max_episode_steps, scenario_name, solver_backend = _resolve_env_settings_3d(args, metadata)
        contact_backend = args.contact_backend
    else:
        observation_config, max_episode_steps, scenario_name, contact_backend = _resolve_env_settings(args, metadata)
        solver_backend = args.solver_backend
    if metadata is None and task == "arm2d":
        observation_config = _infer_observation_config_from_model(model)
    render_mode = None if args.play_render_mode == "none" else args.play_render_mode
    env = _make_direct_env(
        task=task,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        observation_config=observation_config,
        contact_backend=contact_backend,
        solver_backend=solver_backend,
    )
    reset_options = _make_reset_options(scenario_name)
    try:
        for episode in range(args.play_episodes):
            obs, info = env.reset(seed=args.seed + episode, options=reset_options)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if render_mode == "rgb_array":
                    env.render()
                if terminated or truncated:
                    orientation = (
                        info.get("orientation_error_to_target")
                        if task == "arm3d"
                        else info.get("angle_error_to_target")
                    )
                    print(
                        f"Episode {episode + 1} finished. "
                        f"Target Reached: {info['success']} Distance: {info['distance_to_target']:.3f} "
                        f"Orientation: {float(orientation):.3f}"
                    )
                    break
    finally:
        env.close()


def _play_model_based_cem(args, artifact_path: str, metadata: dict | None) -> None:
    print(f"Loading model-based controller from {artifact_path}...")
    task = _resolve_task(args, metadata)
    if task == "arm3d":
        observation_config, max_episode_steps, scenario_name, solver_backend = _resolve_env_settings_3d(args, metadata)
        contact_backend = args.contact_backend
    else:
        observation_config, max_episode_steps, scenario_name, contact_backend = _resolve_env_settings(args, metadata)
        solver_backend = args.solver_backend
    render_mode = None if args.play_render_mode == "none" else args.play_render_mode
    env = _make_direct_env(
        task=task,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        observation_config=observation_config,
        contact_backend=contact_backend,
        solver_backend=solver_backend,
    )
    if task == "arm3d":
        controller = ModelBasedCEM3DController.load(
            artifact_path,
            sim_template=env.sim,
            control_config=env.control_config,
            reward_config=env.reward_config,
            seed=args.seed,
        )
    else:
        controller = ModelBasedCEMController.load(
            artifact_path,
            sim_template=env.sim,
            control_config=env.control_config,
            reward_config=env.reward_config,
            seed=args.seed,
        )
    reset_options = _make_reset_options(scenario_name)
    try:
        for episode in range(args.play_episodes):
            _, info = env.reset(seed=args.seed + episode, options=reset_options)
            controller.reset()
            while True:
                action = controller.act(env.sim)
                _, reward, terminated, truncated, info = env.step(action)
                if render_mode == "rgb_array":
                    env.render()
                if terminated or truncated:
                    plan_summary = controller.episode_optimization_event_summary()
                    orientation = (
                        info.get("orientation_error_to_target")
                        if task == "arm3d"
                        else info.get("angle_error_to_target")
                    )
                    print(
                        f"Episode {episode + 1} finished. "
                        f"Target Reached: {info['success']} Distance: {info['distance_to_target']:.3f} "
                        f"Orientation: {float(orientation):.3f} "
                        f"PlannerEvents: {int(plan_summary['event_count'])} "
                        f"PlannerPenetrations: {int(plan_summary['penetrating_shape_count'])} "
                        f"PlannerWorstClearance: {plan_summary['worst_min_clearance']:.3e}"
                    )
                    break
    finally:
        env.close()


def play(args):
    register_soft_arm_2d_env()
    register_soft_arm_3d_env()
    artifact_kind, artifact_path = _resolve_model_artifact(args.model_path)
    metadata = _load_run_metadata(args.model_path)
    if artifact_kind == "ppo":
        _play_ppo(args, artifact_path, metadata)
        return
    if artifact_kind == "model_based_cem":
        _play_model_based_cem(args, artifact_path, metadata)
        return
    raise ValueError(f"Unsupported artifact type {artifact_kind!r}.")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.play:
        play(args)
    else:
        train(args)
