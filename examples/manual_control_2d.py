from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tasks.arm2d import SoftArmObstacle2DEnv

def main() -> None:
    env = SoftArmObstacle2DEnv(render_mode="human", max_steps=500)
    observation, info = env.reset(seed=0)
    current_action = np.zeros(env.action_space.shape, dtype=np.float32)
    suppress_callbacks = False

    # 加高画板以为更多拉线滑块腾出空间
    figure, axis = plt.subplots(figsize=(7, 4.5))
    plt.subplots_adjust(bottom=0.45)
    axis.set_title("Soft Arm 2D (Tendon-Driven Control)")
    axis.set_xticks([])
    axis.set_yticks([])

    sliders: list[Slider] = []
    labels = env.action_labels
    for index, label in enumerate(labels):
        slider_axis = figure.add_axes([0.18, 0.35 - index * 0.06, 0.62, 0.04])
        slider = Slider(slider_axis, label, -1.0, 1.0, valinit=0.0)
        sliders.append(slider)

    reset_axis = figure.add_axes([0.18, 0.04, 0.18, 0.06])
    reset_button = Button(reset_axis, "Reset")

    def step_env() -> None:
        nonlocal observation, info
        observation, reward, terminated, truncated, info = env.step(current_action)
        print(
            f"tip={info['tip_position']} target={info['target_position']} "
            f"reward={reward:.3f} collisions={info['collision_count']} "
            f"state={info['contact_summary']['contact_state']} "
            f"clearance={info['contact_summary']['min_clearance']:.3f} "
            f"path={info['evaluation_metrics']['path_length']:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            observation, info = env.reset()

    def on_slider_change(_: float) -> None:
        if suppress_callbacks:
            return
        for index, slider in enumerate(sliders):
            current_action[index] = slider.val
        step_env()

    def on_reset(_: object) -> None:
        nonlocal suppress_callbacks
        nonlocal observation, info
        observation, info = env.reset()
        current_action[:] = 0.0
        suppress_callbacks = True
        for slider in sliders:
            slider.set_val(0.0)
        suppress_callbacks = False

    for slider in sliders:
        slider.on_changed(on_slider_change)
    reset_button.on_clicked(on_reset)

    print("Close the slider window to exit.")
    try:
        plt.show(block=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()
