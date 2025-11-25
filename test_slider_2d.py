#!/usr/bin/env python3
# test_slider_2d.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import gymnasium as gym
import sys
import time

# ---------------- 环境 ---------------- #
from tasks.arm2d.env_2d import SoftArmObstacle2DPyEnv as Env
env = Env(render_mode="human")
obs, _ = env.reset()

# 键盘微调步长
STEP = 0.05
current_action = np.zeros(3, dtype=np.float32)

# ---------------- 滑块窗口 ---------------- #
fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(bottom=0.35)
ax.set_title("SoftArm 2D test with Slider Control", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

# 当前离散动作值（-1/0/1）
current_action = np.array([0, 0, 0], dtype=np.int32)

# ---------- 离散滑块：步长=1，刻度=-1,0,1 ---------- #
sliders = []
for i in range(3):
    ax_s = fig.add_axes([0.15, 0.25 - i * 0.07, 0.65, 0.05])
    s = Slider(ax_s, f'action[{i}]', -1, 1, valinit=0, valstep=1)
    sliders.append(s)

# ---------- 执行按钮 ---------- #
def on_execute(_):
    """按钮回调：把当前滑块值发出去"""
    a = np.array([int(s.val) for s in sliders], dtype=np.int32)
    # 离散动作 0~26 编码
    act = (a[0] + 1) * 9 + (a[1] + 1) * 3 + (a[2] + 1)
    obs, rew, _, _, _ = env.step(act)
    # print(f"执行动作 {a} -> 离散编号 {act}   末端={obs[:2]}  reward={rew:.3f}")

ax_btn = fig.add_axes([0.8, 0.05, 0.15, 0.08])
btn = Button(ax_btn, "action")
btn.on_clicked(on_execute)

# ---------- 空格键也能执行 ---------- #
def on_key(event):
    if event.key == ' ':
        on_execute(None)

fig.canvas.mpl_connect('key_press_event', on_key)

print("Usage:")
print("  Drag sliders to -1, 0, 1")
print("  Click 'Execute' or press SPACE to send action")
print("  Ctrl+C to quit")


# ---------------- 主循环 ---------------- #
try:
    plt.show()
except KeyboardInterrupt:
    pass

env.close()
print("测试结束。")