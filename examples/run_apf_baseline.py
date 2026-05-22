from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tasks.arm2d import (
    APFController2D,
    APFController2DConfig,
    Scenario2D,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
    SoftArmSim2DConfig,
    TendonControl2DConfig,
    get_preset_scenario,
)
from tasks.arm2d.apf_controller import sample_tip_space_simplified_apf_grid


SCENARIO_LIBRARY: dict[str, dict[str, object]] = {
    "baseline": {
        "group": "success",
        "description": "Single obstacle baseline for quick smoke testing.",
        "scenario": Scenario2D(
            name="baseline",
            obstacles=[(11.0, 3.0, 1.0)],
            target=(22.6, 5.2),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "multi_guard_4a": {
        "group": "success",
        "description": "Four obstacles with mixed upper/lower guards away from the nominal path.",
        "scenario": Scenario2D(
            name="multi_guard_4a",
            obstacles=[(11.0, -2.0, 1.0), (15.73, -3.41, 0.57), (10.15, 5.83, 0.92), (15.46, 4.4, 0.77)],
            target=(22.6, 6.2),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "multi_guard_4b": {
        "group": "success",
        "description": "Four obstacles with wider workspace clutter and a clear but nontrivial corridor.",
        "scenario": Scenario2D(
            name="multi_guard_4b",
            obstacles=[(11.0, -2.0, 1.0), (18.42, 5.87, 0.55), (17.72, -7.43, 0.84), (11.58, 6.67, 0.77)],
            target=(19.6, 5.2),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "multi_guard_5": {
        "group": "success",
        "description": "Five obstacles with denser lower-half clutter that still admits a safe APF path.",
        "scenario": Scenario2D(
            name="multi_guard_5",
            obstacles=[(11.0, -2.0, 1.0), (10.76, 6.15, 0.86), (13.03, -5.45, 0.73), (17.17, -4.08, 0.57), (15.22, -2.92, 0.82)],
            target=(19.6, 5.2),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "channel_challenge": {
        "group": "challenge",
        "description": "Narrow alternating corridor inside the reachable workspace; useful for exposing local-minimum stalls.",
        "scenario": Scenario2D(
            name="channel_challenge",
            obstacles=[(10.5, 4.0, 1.1), (12.2, -3.0, 1.0), (14.5, 3.6, 0.95), (16.5, -2.0, 0.9), (18.0, 3.1, 0.75)],
            target=(18.6, 6.5),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "dense_challenge": {
        "group": "challenge",
        "description": "Dense mixed-side clutter near the approach corridor for stress-testing clearance management.",
        "scenario": Scenario2D(
            name="dense_challenge",
            obstacles=[(10.6, -1.6, 0.95), (12.4, 3.8, 0.9), (14.1, -2.8, 0.85), (15.8, 3.2, 0.8), (16.8, -1.6, 0.6), (18.0, 2.8, 0.6)],
            target=(22.5, 5.0),
            target_angle=0.0,
            base_x=0.0,
            kappas=np.zeros(2, dtype=float),
        ),
    },
    "contact_left": {
        "group": "challenge",
        "description": "Existing hard preset with staggered obstacles along the approach path.",
        "scenario": get_preset_scenario("contact_left"),
    },
    "contact_channel": {
        "group": "challenge",
        "description": "Existing hard preset with a narrow channel that exposes APF local-minimum behavior.",
        "scenario": get_preset_scenario("contact_channel"),
    },
}


DEFAULT_CONTROLLER_CONFIG = APFController2DConfig(
    attractive_gain=0.35,
    repulsive_gain=0.04,
    safe_threshold=3.0,
    action_gain=1.0,
    finite_difference_epsilon=1e-4,
    backbone_samples_per_segment=9,
    min_clearance=1e-2,
    max_repulsive_force=4.0,
    max_total_force=8.0,
    base_movement_weight=0.5,
    damping_lambda=0.01,
)

DEFAULT_POTENTIAL_RESOLUTION = 180
DEFAULT_POTENTIAL_ALPHA = 0.42
DEFAULT_POTENTIAL_CONTOURS = 14
DEFAULT_FORCE_FIELD_STRIDE = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run APF baselines on single or multiple 2D obstacle scenarios.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="human")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--hold-final-seconds", type=float, default=2.0)
    parser.add_argument("--scenario-name", choices=sorted(SCENARIO_LIBRARY.keys()), default="baseline")
    parser.add_argument("--scenario-set", choices=["single", "success", "challenge", "all"], default="single")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--show-potential", action="store_true")
    parser.add_argument("--potential-resolution", type=int, default=DEFAULT_POTENTIAL_RESOLUTION)
    parser.add_argument("--potential-alpha", type=float, default=DEFAULT_POTENTIAL_ALPHA)
    parser.add_argument("--potential-contours", type=int, default=DEFAULT_POTENTIAL_CONTOURS)
    parser.add_argument("--show-force-field", action="store_true")
    parser.add_argument("--force-field-stride", type=int, default=DEFAULT_FORCE_FIELD_STRIDE)
    return parser


def _clone_scenario(scenario: Scenario2D) -> Scenario2D:
    return Scenario2D(
        name=scenario.name,
        obstacles=None if scenario.obstacles is None else list(scenario.obstacles),
        target=None if scenario.target is None else np.asarray(scenario.target, dtype=float).copy(),
        target_angle=None if scenario.target_angle is None else float(scenario.target_angle),
        base_x=scenario.base_x,
        kappas=None if scenario.kappas is None else np.asarray(scenario.kappas, dtype=float).copy(),
        tendon_lengths=None if scenario.tendon_lengths is None else np.asarray(scenario.tendon_lengths, dtype=float).copy(),
    )


def _round_signature(values: np.ndarray | tuple[float, ...], *, decimals: int = 6) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in np.round(array, decimals=decimals))


def _build_potential_scene_transform(
    env: SoftArmObstacle2DEnv,
    args: argparse.Namespace,
    *,
    controller_config: APFController2DConfig,
):
    potential_cache_key: tuple[object, ...] | None = None
    potential_cache_data: dict[str, object] | None = None

    def transform(scene: dict) -> dict:
        nonlocal potential_cache_key, potential_cache_data

        obstacle_signature = tuple(
            (
                float(np.round(obstacle.center[0], 6)),
                float(np.round(obstacle.center[1], 6)),
                float(np.round(obstacle.radius, 6)),
            )
            for obstacle in env.sim.obstacles
        )
        cache_key = (
            _round_signature(env.render_config.xlim),
            _round_signature(env.render_config.ylim),
            int(args.potential_resolution),
            bool(args.show_force_field),
            _round_signature(env.sim.target),
            obstacle_signature,
            float(controller_config.attractive_gain),
            float(controller_config.repulsive_gain),
            float(controller_config.safe_threshold),
            float(controller_config.min_clearance),
        )
        if potential_cache_key != cache_key or potential_cache_data is None:
            # This background is a tip-space simplified APF visualization. It is
            # intentionally not the full continuum-arm configuration-space potential.
            sampled = sample_tip_space_simplified_apf_grid(
                env.render_config.xlim,
                env.render_config.ylim,
                args.potential_resolution,
                target=env.sim.target,
                obstacles=env.sim.obstacles,
                attractive_gain=controller_config.attractive_gain,
                repulsive_gain=controller_config.repulsive_gain,
                safe_threshold=controller_config.safe_threshold,
                min_clearance=controller_config.min_clearance,
                return_force_field=bool(args.show_force_field),
            )
            if args.show_force_field:
                X, Y, U, Fx, Fy = sampled
            else:
                X, Y, U = sampled
                Fx = None
                Fy = None

            potential_cache_data = {
                "X": X,
                "Y": Y,
                "U": U,
                "alpha": args.potential_alpha,
                "contours": args.potential_contours,
                "cmap": "cividis",
                "contour_color": "#334155",
                "contour_alpha": 0.5,
                "show_force_field": bool(args.show_force_field),
                "force_field_stride": args.force_field_stride,
                "force_field_color": "#0f172a",
                "force_field_alpha": 0.42,
            }
            if Fx is not None and Fy is not None:
                potential_cache_data["Fx"] = Fx
                potential_cache_data["Fy"] = Fy
            potential_cache_key = cache_key

        decorated_scene = dict(scene)
        decorated_scene["potential_field"] = potential_cache_data
        return decorated_scene

    return transform


def print_scenarios() -> None:
    print("Available APF scenarios:")
    for name, payload in SCENARIO_LIBRARY.items():
        scenario = payload["scenario"]
        obstacles = scenario.obstacles or []
        target = np.asarray(scenario.target, dtype=float)
        print(
            f"  {name:16s}",
            f"group={payload['group']:9s}",
            f"obstacles={len(obstacles)}",
            f"target=({target[0]:.1f}, {target[1]:.1f})",
            f"- {payload['description']}",
        )


def resolve_scenarios(args: argparse.Namespace) -> list[tuple[str, str, str, Scenario2D]]:
    if args.scenario_set == "single":
        payload = SCENARIO_LIBRARY[args.scenario_name]
        return [(args.scenario_name, payload["group"], payload["description"], _clone_scenario(payload["scenario"]))]

    selected: list[tuple[str, str, str, Scenario2D]] = []
    for name, payload in SCENARIO_LIBRARY.items():
        if args.scenario_set != "all" and payload["group"] != args.scenario_set:
            continue
        selected.append((name, payload["group"], payload["description"], _clone_scenario(payload["scenario"])))
    return selected


def validate_scenario_selection(
    scenarios: list[tuple[str, str, str, Scenario2D]],
    *,
    max_reach: float,
    target_radius: float,
    min_target_gap: float,
) -> list[tuple[str, str, str, Scenario2D]]:
    valid: list[tuple[str, str, str, Scenario2D]] = []
    for name, group, description, scenario in scenarios:
        target = None if scenario.target is None else np.asarray(scenario.target, dtype=float)
        if target is None:
            print(f"Skipping scenario {name}: target is not defined.")
            continue
        if float(np.linalg.norm(target)) > max_reach - target_radius:
            print(
                f"Skipping scenario {name}: target {target} is outside the current reachable workspace "
                f"(max norm {max_reach - target_radius:.2f})."
            )
            continue
        blocked = False
        for obstacle_index, obstacle in enumerate(scenario.obstacles or []):
            x, y, radius = obstacle if not hasattr(obstacle, "center") else (obstacle.center[0], obstacle.center[1], obstacle.radius)
            if float(np.linalg.norm(target - np.array([x, y], dtype=float))) < float(radius) + target_radius + min_target_gap:
                print(
                    f"Skipping scenario {name}: target {target} overlaps obstacle {obstacle_index} "
                    f"under the current target-gap rule."
                )
                blocked = True
                break
        if blocked:
            continue
        valid.append((name, group, description, scenario))
    return valid


def build_env_kwargs(*, scenario: Scenario2D, max_steps: int, render_mode: str | None = None) -> dict[str, object]:
    env_config = SoftArmObstacle2DEnvConfig(
        control=TendonControl2DConfig(
            base_delta_scale=1.0,
            tendon_delta_scale=1.0,
            allow_negative_base_delta=True,
        ),
    )
    sim_config = SoftArmSim2DConfig(
        obstacle_count=len(scenario.obstacles or []),
        target_radius=0.45,
        target_angle_tolerance=float(np.pi),
        min_obstacle_gap=0.5,
        min_target_gap=1.0,
    )
    return {
        "render_mode": render_mode,
        "max_steps": max_steps,
        "env_config": env_config,
        "sim_config": sim_config,
    }


def build_env(*, scenario: Scenario2D, render_mode: str | None, max_steps: int) -> SoftArmObstacle2DEnv:
    return SoftArmObstacle2DEnv(**build_env_kwargs(scenario=scenario, max_steps=max_steps, render_mode=render_mode))


def run_episode(
    *,
    scenario_name: str,
    scenario_group: str,
    scenario_description: str,
    scenario: Scenario2D,
    args: argparse.Namespace,
    render_mode: str | None,
) -> dict[str, object]:
    env = build_env(scenario=scenario, render_mode=render_mode, max_steps=args.max_steps)
    if args.show_potential and render_mode is not None:
        env.set_render_scene_transform(
            _build_potential_scene_transform(
                env,
                args,
                controller_config=DEFAULT_CONTROLLER_CONFIG,
            )
        )
    controller = APFController2D(
        control_config=env.control_config,
        config=DEFAULT_CONTROLLER_CONFIG,
    )

    collision_steps = 0
    total_collision_count = 0
    action_blocked_steps = 0
    min_clearance = float("inf")

    try:
        _, info = env.reset(seed=args.seed, options={"scenario": scenario})
        controller.reset()
        if render_mode == "human":
            env.render()

        print(
            "Starting APF scenario:",
            f"name={scenario_name}",
            f"group={scenario_group}",
            f"obstacles={len(scenario.obstacles or [])}",
            f"target={np.asarray(info['target_position'], dtype=float)}",
        )
        print(f"Description: {scenario_description}")

        terminated = False
        truncated = False
        for step in range(args.max_steps):
            action = controller.act(env.sim)
            _, _, terminated, truncated, info = env.step(action)
            collision_steps += int(info["collision_count"] > 0)
            total_collision_count += int(info["collision_count"])
            action_blocked_steps += int(bool(info["action_blocked"]))
            min_clearance = min(min_clearance, float(info["contact_summary"]["min_clearance"]))

            if step % max(args.log_every, 1) == 0 or terminated or truncated:
                total_force = np.asarray(controller.last_control_info["total_force"], dtype=float)
                print(
                    f"[{scenario_name}]",
                    f"step={step:03d}",
                    f"distance={info['distance_to_target']:.3f}",
                    f"clearance={info['contact_summary']['min_clearance']:.3f}",
                    f"collisions={info['collision_count']}",
                    f"action_blocked={info['action_blocked']}",
                    f"base_delta={float(controller.last_control_info.get('applied_base_delta', 0.0)):.3f}",
                    f"force_norm={np.linalg.norm(total_force):.3f}",
                )

            if render_mode == "rgb_array":
                env.render()
            elif render_mode == "human" and args.fps > 0.0:
                time.sleep(1.0 / float(args.fps))

            if terminated or truncated:
                break

        if render_mode == "human":
            env.render()
            if args.hold_final_seconds > 0.0:
                time.sleep(float(args.hold_final_seconds))

        result = {
            "name": scenario_name,
            "group": scenario_group,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info["success"]),
            "final_distance": float(info["distance_to_target"]),
            "collision_steps": int(collision_steps),
            "total_collision_count": int(total_collision_count),
            "action_blocked_steps": int(action_blocked_steps),
            "min_clearance": float(min_clearance),
            "steps_taken": int(step + 1),
        }
        print(
            "APF scenario result:",
            f"name={scenario_name}",
            f"terminated={terminated}",
            f"truncated={truncated}",
            f"success={result['success']}",
            f"distance={result['final_distance']:.3f}",
            f"collision_steps={collision_steps}",
            f"total_collision_count={total_collision_count}",
            f"action_blocked_steps={action_blocked_steps}",
            f"min_clearance={min_clearance:.3f}",
        )
        return result
    finally:
        env.close()


def summarize_results(results: list[dict[str, object]]) -> None:
    print("\nScenario summary:")
    for result in results:
        print(
            f"  {result['name']:16s}",
            f"group={result['group']:9s}",
            f"success={str(bool(result['success']) and bool(result['terminated']) and not bool(result['truncated']) and int(result['action_blocked_steps']) == 0 and int(result['total_collision_count']) == 0 and float(result['min_clearance']) > 0.0):5s}",
            f"distance={float(result['final_distance']):.3f}",
            f"collision_steps={int(result['collision_steps']):3d}",
            f"blocked_steps={int(result['action_blocked_steps']):3d}",
            f"total_collisions={int(result['total_collision_count']):3d}",
            f"min_clearance={float(result['min_clearance']):.3f}",
            f"steps={int(result['steps_taken']):3d}",
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.list_scenarios:
        print_scenarios()
        return 0

    scenarios = resolve_scenarios(args)
    reference_scenario = scenarios[0][3] if scenarios else Scenario2D()
    reference_env = SoftArmObstacle2DEnv(
        **build_env_kwargs(
            scenario=reference_scenario,
            max_steps=args.max_steps,
            render_mode=None,
        )
    )
    try:
        max_reach = float(reference_env.sim.max_reach)
        target_radius = float(reference_env.sim.target_radius)
        min_target_gap = float(reference_env.sim.sim_config.min_target_gap)
    finally:
        reference_env.close()
    scenarios = validate_scenario_selection(
        scenarios,
        max_reach=max_reach,
        target_radius=target_radius,
        min_target_gap=min_target_gap,
    )
    if not scenarios:
        raise ValueError("No scenarios matched the requested selection.")

    multi_run = len(scenarios) > 1
    render_mode = None if args.render_mode == "none" else args.render_mode
    if multi_run and render_mode == "human":
        render_mode = None
        print("Multiple scenarios selected, disabling live window rendering for batch execution.")

    results: list[dict[str, object]] = []
    for scenario_name, scenario_group, scenario_description, scenario in scenarios:
        results.append(
            run_episode(
                scenario_name=scenario_name,
                scenario_group=scenario_group,
                scenario_description=scenario_description,
                scenario=scenario,
                args=args,
                render_mode=render_mode,
            )
        )

    summarize_results(results)

    if args.allow_failures:
        return 0

    failures = [
        result
        for result in results
        if not (
            bool(result["success"])
            and bool(result["terminated"])
            and not bool(result["truncated"])
            and int(result["action_blocked_steps"]) == 0
            and int(result["total_collision_count"]) == 0
            and int(result["collision_steps"]) == 0
            and float(result["min_clearance"]) > 0.0
        )
    ]
    if failures:
        names = ", ".join(str(result["name"]) for result in failures)
        raise AssertionError(f"APF evaluation failed on: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
