from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import Render3DConfig


class Renderer3D:
    def __init__(
        self,
        *,
        config: Render3DConfig | None = None,
    ):
        self.config = config or Render3DConfig()
        self.figure = None
        self.axis = None

    def _ensure_canvas(self, human: bool) -> None:
        if self.figure is not None and self.axis is not None:
            return
        self.figure = plt.figure(figsize=self.config.figsize)
        self.axis = self.figure.add_subplot(111, projection="3d")
        if human:
            plt.ion()
            plt.show(block=False)

    def _draw_sphere(self, center: np.ndarray, radii: np.ndarray, *, color: str, edge_color: str, alpha: float) -> None:
        assert self.axis is not None
        u = np.linspace(0.0, 2.0 * np.pi, 20)
        v = np.linspace(0.0, np.pi, 12)
        x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
        self.axis.plot_surface(x, y, z, color=color, edgecolor=edge_color, linewidth=0.2, alpha=alpha)

    def _draw(self, scene: dict) -> None:
        assert self.axis is not None
        axis = self.axis
        axis.clear()
        axis.set_xlim(*self.config.xlim)
        axis.set_ylim(*self.config.ylim)
        axis.set_zlim(*self.config.zlim)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(self.config.title)
        if self.config.show_grid:
            axis.grid(alpha=0.25, linestyle="--")

        free_points = np.asarray(scene["free_points"], dtype=float)
        observed_points = np.asarray(scene["observed_points"], dtype=float)
        if free_points.size:
            axis.plot(
                free_points[:, 0],
                free_points[:, 1],
                free_points[:, 2],
                linestyle="--",
                linewidth=2.0,
                color="#94a3b8",
                label="free shape",
            )
        if observed_points.size:
            axis.plot(
                observed_points[:, 0],
                observed_points[:, 1],
                observed_points[:, 2],
                linewidth=3.0,
                color="#2563eb",
                label="contact-aware shape",
            )
        backbone_points = np.asarray(scene.get("backbone_points", np.zeros((0, 3), dtype=float)), dtype=float)
        if backbone_points.size:
            axis.scatter(
                backbone_points[:, 0],
                backbone_points[:, 1],
                backbone_points[:, 2],
                s=18,
                c="#0f766e",
                marker="x",
                linewidths=1.0,
                label="backbone keypoints",
            )

        for obstacle in scene["obstacles"]:
            self._draw_sphere(
                obstacle.center,
                obstacle.radii,
                color="#f87171",
                edge_color="#7f1d1d",
                alpha=0.45,
            )

        target = np.asarray(scene["target"], dtype=float)
        self._draw_sphere(
            target,
            np.repeat(float(scene["target_radius"]), 3),
            color="#4ade80",
            edge_color="#166534",
            alpha=0.65,
        )

        if scene["collision_points"]:
            collision_points = np.asarray(scene["collision_points"], dtype=float)
            axis.scatter(
                collision_points[:, 0],
                collision_points[:, 1],
                collision_points[:, 2],
                s=42,
                c="#f59e0b",
                edgecolors="black",
                linewidths=0.5,
                label="contact",
            )

        base_position = np.asarray(scene["base_position"], dtype=float)
        tip_position = np.asarray(scene["tip_position"], dtype=float)
        axis.scatter(
            base_position[0],
            base_position[1],
            base_position[2],
            c="black",
            s=50,
            marker="s",
            label="base",
        )
        axis.scatter(
            tip_position[0],
            tip_position[1],
            tip_position[2],
            c="#1d4ed8",
            s=70,
            marker="o",
            label="tip",
        )

        contact_summary = scene.get("contact_summary")
        if contact_summary and np.isfinite(contact_summary["min_clearance"]):
            axis.text2D(
                0.02,
                0.98,
                (
                    f"state={contact_summary.get('contact_state', 'free')} "
                    f"clearance={contact_summary['min_clearance']:.4f} "
                    f"penetration={contact_summary.get('primary_penetration', 0.0):.4f}"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cbd5e1"},
            )
        axis.legend(loc="upper left")
        axis.view_init(elev=22.0, azim=-58.0)

    def render(self, scene: dict, mode: str = "human") -> np.ndarray | None:
        if mode not in {"human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode {mode!r}.")
        human = mode == "human"
        self._ensure_canvas(human=human)
        self._draw(scene)
        assert self.figure is not None
        self.figure.canvas.draw()
        if human:
            self.figure.canvas.flush_events()
            return None
        image = np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy()
        return image

    def close(self) -> None:
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axis = None
