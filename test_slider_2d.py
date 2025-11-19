#!/usr/bin/env python3
# test_slider_2d.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import gymnasium as gym
import sys
import time

# ---------------- 环境 ---------------- #
from tasks.arm2d.env_2d import SoftArmObstacle2DPyEnv as Env
env = Env(render_mode="human")
obs, _ = env.reset()

# ---------------- 滑块窗口 ---------------- #
fig, ax = plt.subplots(figsize=(6, 3))
plt.subplots_adjust(bottom=0.3)
ax.set_title("SoftArm 2D test with Slider Control", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

# 3 个滑块（-1~1）
sliders = []
for i in range(3):
    ax_s = fig.add_axes([0.15, 0.25 - i * 0.07, 0.65, 0.05])
    s = Slider(ax_s, f'action[{i}]', -1, 1, valinit=0.0)
    sliders.append(s)

# 键盘微调步长
STEP = 0.05
current_action = np.zeros(3, dtype=np.float32)

def apply_action():
    global obs
    obs, rew, _, _, _ = env.step(current_action)
    print(f"动作 {current_action}  观测 {obs}  奖励 {rew:.3f}")

def on_change(val):
    """滑块拖动"""
    for i, s in enumerate(sliders):
        current_action[i] = s.val
    apply_action()

for s in sliders:
    s.on_changed(on_change)

# ---------------- 键盘 ± 微调 ---------------- #
def on_key(event):
    if event.key in '123':               # 选择维度
        dim = int(event.key) - 1
        if event.key == '+':             # Shift + 1/2/3 → +
            current_action[dim] = np.clip(current_action[dim] + STEP, -1, 1)
            sliders[dim].set_val(current_action[dim])
        elif event.key == '-':           # Shift + 1/2/3 → -
            current_action[dim] = np.clip(current_action[dim] - STEP, -1, 1)
            sliders[dim].set_val(current_action[dim])
        apply_action()

fig.canvas.mpl_connect('key_press_event', on_key)
print("操作说明：")
print("  拖动滑块 → 实时动作")
print("  Shift+1/2/3 → ±0.05 微调")
print("  Ctrl+C 退出")

# ---------------- 主循环（非阻塞） ---------------- #
try:
    plt.show()
except KeyboardInterrupt:
    pass

env.close()
print("测试结束。")