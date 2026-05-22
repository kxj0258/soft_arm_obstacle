from __future__ import annotations

import argparse
import time

from tasks.arm2d import (
    ArmModel2DConfig,
    ContactResponse2DConfig,
    SoftArmObstacle2DEnv,
    list_preset_scenarios,
)
from tasks.arm3d import (
    ArmModel3DConfig,
    SoftArmObstacle3DEnv,
    list_preset_scenarios as list_preset_scenarios_3d,
)


def _scenario_choices() -> list[str]:
    return sorted(set(list_preset_scenarios()) | set(list_preset_scenarios_3d()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a soft-arm obstacle environment.")
    parser.add_argument("--task", choices=["arm2d", "arm3d"], default="arm2d")
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--scenario-name", choices=_scenario_choices(), default=None)
    parser.add_argument("--contact-backend", choices=["heuristic", "optimization"], default="heuristic")
    parser.add_argument(
        "--solver-backend",
        choices=["pcc3d", "cosserat_static", "pcca_contact_3d", "pinn", "hybrid", "oracle"],
        default="hybrid",
        help="3D TDCR backend; ignored for arm2d.",
    )
    return parser


def _make_env(args, render_mode):
    if args.task == "arm3d":
        return SoftArmObstacle3DEnv(
            render_mode=render_mode,
            max_steps=args.steps,
            model_config=ArmModel3DConfig(solver_backend=args.solver_backend),
        )
    return SoftArmObstacle2DEnv(
        render_mode=render_mode,
        max_steps=args.steps,
        model_config=ArmModel2DConfig(
            contact_response=ContactResponse2DConfig(backend=args.contact_backend),
        ),
    )


def _print_episode_summary(*, task: str, episode: int, step: int, episode_return: float, info: dict) -> None:
    backend_reason = info.get("contact_backend_details", {}).get("reason", "")
    orientation_key = "orientation_error_to_target" if task == "arm3d" else "angle_error_to_target"
    orientation_label = "orientation_error" if task == "arm3d" else "angle_error"
    print(
        f"episode={episode + 1} steps={step + 1} "
        f"task={task} "
        f"return={episode_return:.3f} distance={info['distance_to_target']:.3f} "
        f"{orientation_label}={info[orientation_key]:.3f} "
        f"collisions={info['collision_count']} "
        f"backend={info['contact_backend']} "
        f"backend_reason={backend_reason} "
        f"state={info['contact_summary']['contact_state']} "
        f"clearance={info['contact_summary']['min_clearance']:.3f} "
        f"path={info['evaluation_metrics']['path_length']:.3f} "
        f"success={info['success']}"
    )


def main() -> int:
    args = build_parser().parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = _make_env(args, render_mode)

    try:
        for episode in range(args.episodes):
            reset_options = None if args.scenario_name is None else {"scenario_name": args.scenario_name}
            observation, info = env.reset(seed=args.seed + episode, options=reset_options)
            episode_return = 0.0

            for step in range(args.steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward

                if render_mode == "rgb_array":
                    env.render()
                elif render_mode == "human" and args.sleep > 0.0:
                    time.sleep(args.sleep)

                if terminated or truncated:
                    break

            _print_episode_summary(
                task=args.task,
                episode=episode,
                step=step,
                episode_return=episode_return,
                info=info,
            )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
