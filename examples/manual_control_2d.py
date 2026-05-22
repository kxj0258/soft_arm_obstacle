from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tasks.arm2d import (
    APFController2DConfig,
    ArmModel2DConfig,
    ContactResponse2DConfig,
    SoftArmObstacle2DEnv,
    SoftArmObstacle2DEnvConfig,
    TendonControl2DConfig,
    list_preset_scenarios,
)
from tasks.arm2d.apf_controller import sample_tip_space_simplified_apf_grid


DEFAULT_FPS = 30
DEFAULT_SCENARIO = "contact_left"
DEFAULT_POTENTIAL_RESOLUTION = 180
DEFAULT_POTENTIAL_ALPHA = 0.42
DEFAULT_POTENTIAL_CONTOURS = 14
DEFAULT_FORCE_FIELD_STRIDE = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive 2D manual contact debugger.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--scenario-name", choices=list_preset_scenarios(), default=DEFAULT_SCENARIO)
    parser.add_argument("--contact-backend", choices=["heuristic", "optimization"], default="heuristic")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show-potential", action="store_true")
    parser.add_argument("--potential-resolution", type=int, default=DEFAULT_POTENTIAL_RESOLUTION)
    parser.add_argument("--potential-alpha", type=float, default=DEFAULT_POTENTIAL_ALPHA)
    parser.add_argument("--potential-contours", type=int, default=DEFAULT_POTENTIAL_CONTOURS)
    parser.add_argument("--show-force-field", action="store_true")
    parser.add_argument("--force-field-stride", type=int, default=DEFAULT_FORCE_FIELD_STRIDE)
    return parser


def _format_bool(value: bool) -> str:
    return "YES" if value else "NO"


def _build_action(env: SoftArmObstacle2DEnv, pressed: pygame.key.ScancodeWrapper) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    if pressed[pygame.K_RIGHT]:
        action[0] += 1.0
    if pressed[pygame.K_LEFT]:
        action[0] -= 1.0

    if env.num_kappas >= 1:
        if pressed[pygame.K_w]:
            action[1] -= 1.0
            action[2] += 1.0
        if pressed[pygame.K_s]:
            action[1] += 1.0
            action[2] -= 1.0

    if env.num_kappas >= 2:
        if pressed[pygame.K_a]:
            action[3] += 1.0
            action[4] -= 1.0
        if pressed[pygame.K_d]:
            action[3] -= 1.0
            action[4] += 1.0

    return np.clip(action, env.action_space.low, env.action_space.high)


def _blit_frame(surface: pygame.Surface, frame: np.ndarray) -> pygame.Surface:
    rgb_frame = np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
    return pygame.surfarray.make_surface(rgb_frame)


def _round_signature(values: np.ndarray | tuple[float, ...], *, decimals: int = 6) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in np.round(array, decimals=decimals))


def _draw_telemetry(
    surface: pygame.Surface,
    *,
    font: pygame.font.Font,
    info: dict,
    fps: float,
    scenario_name: str,
    backend: str,
) -> None:
    telemetry_lines = [
        f"Scenario: {scenario_name}",
        f"Backend: {backend}",
        f"FPS: {fps:.1f}",
        "Kappas: [" + ", ".join(f"{float(kappa): .3f}" for kappa in np.asarray(info["kappas"], dtype=float)) + "]",
        f"Base X: {info['base_x']:.3f}",
        f"Action Blocked: {_format_bool(bool(info.get('action_blocked', False)))}",
        f"Contact: {info['contact_summary']['contact_state']}",
        f"Clearance: {info['contact_summary']['min_clearance']:.3f}",
    ]

    line_height = font.get_linesize()
    box_height = 18 + line_height * len(telemetry_lines)
    box_width = 320
    overlay = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    overlay.fill((255, 255, 255, 200))
    surface.blit(overlay, (12, 12))

    for index, line in enumerate(telemetry_lines):
        text_surface = font.render(line, True, (15, 23, 42))
        surface.blit(text_surface, (24, 22 + index * line_height))

    help_lines = [
        "Hold W/S: tendon 1",
        "Hold A/D: tendon 2",
        "Hold Left/Right: base X",
        "Press R: reset, Esc: quit",
    ]
    help_overlay_height = 14 + line_height * len(help_lines)
    help_overlay = pygame.Surface((260, help_overlay_height), pygame.SRCALPHA)
    help_overlay.fill((15, 23, 42, 175))
    help_x = 12
    help_y = surface.get_height() - help_overlay_height - 12
    surface.blit(help_overlay, (help_x, help_y))
    for index, line in enumerate(help_lines):
        text_surface = font.render(line, True, (241, 245, 249))
        surface.blit(text_surface, (help_x + 12, help_y + 8 + index * line_height))


def main() -> int:
    args = build_parser().parse_args()
    env = SoftArmObstacle2DEnv(
        render_mode="rgb_array",
        max_steps=args.max_steps,
        model_config=ArmModel2DConfig(
            contact_response=ContactResponse2DConfig(backend=args.contact_backend),
        ),
        env_config=SoftArmObstacle2DEnvConfig(
            control=TendonControl2DConfig(
                base_delta_scale=10.0 / max(args.fps, 1),
                tendon_delta_scale=1.0 / max(args.fps, 1),
                allow_negative_base_delta=True,
            ),
        ),
    )

    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    potential_config = APFController2DConfig()

    observation = None
    info = {}
    frame_surface: pygame.Surface | None = None
    screen: pygame.Surface | None = None
    potential_cache_key: tuple[object, ...] | None = None
    potential_cache_data: dict[str, object] | None = None

    def reset_env(seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, reset_info = env.reset(seed=seed, options={"scenario_name": args.scenario_name})
        return obs, reset_info

    def build_potential_field_scene() -> dict[str, object] | None:
        nonlocal potential_cache_key, potential_cache_data
        if not args.show_potential:
            return None

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
            float(potential_config.attractive_gain),
            float(potential_config.repulsive_gain),
            float(potential_config.safe_threshold),
            float(potential_config.min_clearance),
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
                attractive_gain=potential_config.attractive_gain,
                repulsive_gain=potential_config.repulsive_gain,
                safe_threshold=potential_config.safe_threshold,
                min_clearance=potential_config.min_clearance,
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
        return potential_cache_data

    def render_frame() -> np.ndarray:
        if not args.show_potential:
            frame = env.render()
        else:
            scene = env.sim.scene_state()
            potential_field = build_potential_field_scene()
            if potential_field is not None:
                scene["potential_field"] = potential_field
            frame = env.render(scene=scene)
        if frame is None:
            raise RuntimeError("Expected an rgb_array frame from the 2D renderer.")
        return frame

    try:
        observation, info = reset_env(seed=args.seed)
        frame = render_frame()
        frame_surface = _blit_frame(pygame.Surface((frame.shape[1], frame.shape[0])), frame)
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption("Soft Arm 2D Manual Debugger")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = reset_env()

            pressed = pygame.key.get_pressed()
            action = _build_action(env, pressed)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = reset_env()

            frame = render_frame()
            frame_surface = _blit_frame(screen, frame)
            screen.blit(frame_surface, (0, 0))
            _draw_telemetry(
                screen,
                font=font,
                info=info,
                fps=clock.get_fps() or float(args.fps),
                scenario_name=args.scenario_name,
                backend=args.contact_backend,
            )
            pygame.display.flip()
            clock.tick(max(args.fps, 1))
    finally:
        env.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
