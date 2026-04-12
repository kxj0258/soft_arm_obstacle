from __future__ import annotations

from abc import ABC

import gymnasium as gym


class BaseSoftArmEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, max_steps: int = 200):
        super().__init__()
        valid_render_modes = {None, "human", "rgb_array"}
        if render_mode not in valid_render_modes:
            raise ValueError(f"render_mode must be one of {valid_render_modes}, got {render_mode!r}.")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self._elapsed_steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

    def close(self) -> None:
        return None
