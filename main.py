from __future__ import annotations

import argparse
import time

from tasks.arm2d import SoftArmObstacle2DEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pure Python 2D soft-arm obstacle environment.")
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.03)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = SoftArmObstacle2DEnv(render_mode=render_mode, max_steps=args.steps)

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
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

            print(
                f"episode={episode + 1} steps={step + 1} "
                f"return={episode_return:.3f} distance={info['distance_to_target']:.3f} "
                f"collisions={info['collision_count']} success={info['success']}"
            )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
