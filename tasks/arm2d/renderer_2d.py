from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .config import Render2DConfig


class Renderer2D:
    def __init__(
        self,
        *,
        config: Render2DConfig | None = None,
    ):
        self.config = config or Render2DConfig()
        self.figure = None
        self.axis = None

    def _ensure_canvas(self, human: bool) -> None:
        if self.figure is not None and self.axis is not None:
            return
        self.figure, self.axis = plt.subplots(figsize=self.config.figsize)
        if human:
            plt.ion()
            plt.show(block=False)

    def _draw_potential_field(self, scene: dict) -> None:
        potential_field = scene.get("potential_field")
        if not potential_field:
            return

        assert self.axis is not None
        axis = self.axis
        X = np.asarray(potential_field.get("X", np.zeros((0, 0), dtype=float)), dtype=float)
        Y = np.asarray(potential_field.get("Y", np.zeros((0, 0), dtype=float)), dtype=float)
        U = np.asarray(potential_field.get("U", np.zeros((0, 0), dtype=float)), dtype=float)
        if X.ndim != 2 or X.shape != Y.shape or X.shape != U.shape or X.size == 0:
            raise ValueError("potential_field requires matching 2D X, Y, U arrays.")

        finite_mask = np.isfinite(U)
        if not np.any(finite_mask):
            return

        alpha = float(np.clip(potential_field.get("alpha", 0.4), 0.0, 1.0))
        contour_count = max(int(potential_field.get("contours", 12)), 2)
        cmap = str(potential_field.get("cmap", "cividis"))
        contour_color = str(potential_field.get("contour_color", "#334155"))
        contour_linewidth = float(potential_field.get("contour_linewidth", 0.8))
        contour_alpha = float(np.clip(potential_field.get("contour_alpha", 0.55), 0.0, 1.0))
        potential_span = float(np.nanmax(U) - np.nanmin(U))
        if potential_span <= 1e-12:
            return

        axis.contourf(
            X,
            Y,
            U,
            levels=max(contour_count * 2, contour_count + 1),
            cmap=cmap,
            alpha=alpha,
            zorder=0,
        )
        axis.contour(
            X,
            Y,
            U,
            levels=contour_count,
            colors=contour_color,
            linewidths=contour_linewidth,
            alpha=contour_alpha,
            zorder=1,
        )

        show_force_field = bool(potential_field.get("show_force_field", False))
        if not show_force_field:
            return

        Fx = np.asarray(potential_field.get("Fx", np.zeros_like(U)), dtype=float)
        Fy = np.asarray(potential_field.get("Fy", np.zeros_like(U)), dtype=float)
        if Fx.shape != U.shape or Fy.shape != U.shape:
            raise ValueError("potential_field Fx and Fy must match the sampled potential shape.")

        stride = max(int(potential_field.get("force_field_stride", 10)), 1)
        X_sparse = X[::stride, ::stride]
        Y_sparse = Y[::stride, ::stride]
        Fx_sparse = Fx[::stride, ::stride]
        Fy_sparse = Fy[::stride, ::stride]
        norms = np.linalg.norm(np.stack((Fx_sparse, Fy_sparse), axis=-1), axis=-1)
        valid = norms > 1e-8
        if not np.any(valid):
            return

        display_fx = np.zeros_like(Fx_sparse)
        display_fy = np.zeros_like(Fy_sparse)
        display_fx[valid] = Fx_sparse[valid] / norms[valid]
        display_fy[valid] = Fy_sparse[valid] / norms[valid]
        x_spacing = float(np.mean(np.abs(np.diff(X[0, :])))) if X.shape[1] > 1 else 1.0
        y_spacing = float(np.mean(np.abs(np.diff(Y[:, 0])))) if Y.shape[0] > 1 else 1.0
        arrow_length = max(0.55 * stride * min(x_spacing, y_spacing), 1e-6)
        axis.quiver(
            X_sparse,
            Y_sparse,
            display_fx,
            display_fy,
            angles="xy",
            scale_units="xy",
            scale=1.0 / arrow_length,
            pivot="mid",
            color=str(potential_field.get("force_field_color", "#0f172a")),
            alpha=float(np.clip(potential_field.get("force_field_alpha", 0.45), 0.0, 1.0)),
            width=float(potential_field.get("force_field_width", 0.0028)),
            zorder=2,
        )

    def _draw(self, scene: dict) -> None:
        assert self.axis is not None
        axis = self.axis
        axis.clear()
        axis.set_aspect("equal")
        axis.set_xlim(*self.config.xlim)
        axis.set_ylim(*self.config.ylim)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(self.config.title)
        self._draw_potential_field(scene)
        if self.config.show_grid:
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
                zorder=3,
            )
        if len(observed_points):
            axis.plot(
                observed_points[:, 0],
                observed_points[:, 1],
                linewidth=3.0,
                color="#2563eb",
                label="contact-aware shape",
                zorder=4,
            )
        backbone_points = np.asarray(scene.get("backbone_points", np.zeros((0, 2), dtype=float)), dtype=float)
        if backbone_points.size:
            axis.scatter(
                backbone_points[:, 0],
                backbone_points[:, 1],
                s=18,
                c="#0f766e",
                marker="x",
                linewidths=1.0,
                label="backbone keypoints",
                zorder=4,
            )

        for obstacle in scene["obstacles"]:
            axis.add_patch(
                Circle(
                    obstacle.center,
                    obstacle.radius,
                    facecolor="#f87171",
                    edgecolor="#7f1d1d",
                    alpha=0.7,
                    zorder=3,
                )
            )

        axis.add_patch(
            Circle(
                scene["target"],
                scene["target_radius"],
                facecolor="#4ade80",
                edgecolor="#166534",
                alpha=0.8,
                zorder=4,
            )
        )
        target_direction = np.array(
            [np.cos(float(scene.get("target_angle", 0.0))), np.sin(float(scene.get("target_angle", 0.0)))],
            dtype=float,
        )
        target_arrow_length = max(float(scene["target_radius"]) * 1.8, 0.5)
        axis.arrow(
            float(scene["target"][0]),
            float(scene["target"][1]),
            float(target_direction[0] * target_arrow_length),
            float(target_direction[1] * target_arrow_length),
            width=0.06,
            head_width=0.4,
            head_length=0.5,
            length_includes_head=True,
            color="#166534",
            alpha=0.95,
            zorder=4,
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

        contact_markers = scene.get("contact_markers", [])
        if contact_markers:
            x_span = float(self.config.xlim[1] - self.config.xlim[0])
            y_span = float(self.config.ylim[1] - self.config.ylim[0])
            base_arrow_length = 0.045 * min(x_span, y_span)
            for index, marker in enumerate(contact_markers):
                point = np.asarray(marker.get("point", np.zeros(2, dtype=float)), dtype=float).reshape(2)
                normal = np.asarray(marker.get("normal", np.array([1.0, 0.0], dtype=float)), dtype=float).reshape(2)
                normal_norm = float(np.linalg.norm(normal))
                if not np.isfinite(normal_norm) or normal_norm < 1e-8:
                    continue
                unit_normal = normal / normal_norm
                magnitude = float(marker.get("magnitude", 0.0))
                arrow_length = base_arrow_length * (1.0 + min(max(magnitude, 0.0), 1.0))
                axis.scatter(
                    point[0],
                    point[1],
                    s=28,
                    c="#dc2626",
                    marker="o",
                    linewidths=0.0,
                    zorder=6,
                    label="contact normal" if index == 0 else None,
                )
                axis.arrow(
                    float(point[0]),
                    float(point[1]),
                    float(unit_normal[0] * arrow_length),
                    float(unit_normal[1] * arrow_length),
                    width=0.045,
                    head_width=0.32,
                    head_length=0.38,
                    length_includes_head=True,
                    color="#dc2626",
                    alpha=0.95,
                    zorder=6,
                )

        base_position = scene["base_position"]
        tip_position = scene["tip_position"]
        axis.scatter(base_position[0], base_position[1], c="black", s=50, marker="s", label="base", zorder=6)
        axis.scatter(tip_position[0], tip_position[1], c="#1d4ed8", s=70, marker="o", label="tip", zorder=6)
        contact_summary = scene.get("contact_summary")
        if contact_summary and np.isfinite(contact_summary["min_clearance"]):
            clearance_text = (
                f"state={contact_summary.get('contact_state', 'free')} "
                f"clearance={contact_summary['min_clearance']:.2f} "
                f"penetration={contact_summary.get('primary_penetration', 0.0):.2f}"
            )
            axis.text(
                0.02,
                0.98,
                clearance_text,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cbd5e1"},
            )
        axis.legend(loc="upper left")

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
