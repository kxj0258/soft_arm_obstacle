from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


class Renderer2D:
    def __init__(
        self,
        *,
        xlim: tuple[float, float] = (-2.0, 42.0),
        ylim: tuple[float, float] = (-20.0, 20.0),
        figsize: tuple[float, float] = (8.0, 5.0),
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        self.figure = None
        self.axis = None

    def _ensure_canvas(self, human: bool) -> None:
        if self.figure is not None and self.axis is not None:
            return
        self.figure, self.axis = plt.subplots(figsize=self.figsize)
        if human:
            plt.ion()
            plt.show(block=False)

    def _draw(self, scene: dict) -> None:
        assert self.axis is not None
        axis = self.axis
        axis.clear()
        axis.set_aspect("equal")
        axis.set_xlim(*self.xlim)
        axis.set_ylim(*self.ylim)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title("Soft Arm Obstacle 2D")
        axis.grid(alpha=0.2, linestyle="--")

        free_points = scene["free_points"]
        observed_points = scene["observed_points"]
        if len(free_points):
            axis.plot(
                free_points[:, 0],
                free_points[:, 1],
                linestyle="--",
                linewidth=2.0,
                color="#94a3b8",
                label="free shape",
            )
        if len(observed_points):
            axis.plot(
                observed_points[:, 0],
                observed_points[:, 1],
                linewidth=3.0,
                color="#2563eb",
                label="contact-aware shape",
            )

        for obstacle in scene["obstacles"]:
            axis.add_patch(
                Circle(
                    obstacle.center,
                    obstacle.radius,
                    facecolor="#f87171",
                    edgecolor="#7f1d1d",
                    alpha=0.7,
                )
            )

        axis.add_patch(
            Circle(
                scene["target"],
                scene["target_radius"],
                facecolor="#4ade80",
                edgecolor="#166534",
                alpha=0.8,
            )
        )

        if scene["collision_points"]:
            collision_points = np.asarray(scene["collision_points"], dtype=float)
            axis.scatter(
                collision_points[:, 0],
                collision_points[:, 1],
                s=40,
                c="#f59e0b",
                edgecolors="black",
                linewidths=0.5,
                label="contact",
                zorder=5,
            )

        base_position = scene["base_position"]
        tip_position = scene["tip_position"]
        axis.scatter(base_position[0], base_position[1], c="black", s=50, marker="s", label="base", zorder=6)
        axis.scatter(tip_position[0], tip_position[1], c="#1d4ed8", s=70, marker="o", label="tip", zorder=6)
        axis.legend(loc="upper left")

    def render(self, scene: dict, mode: str = "human") -> np.ndarray | None:
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
