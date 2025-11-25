import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle


class Renderer2D:
    def __init__(self, fps=30):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_aspect('equal')
        self.ax_xlim = (-5.0, 55.0)
        self.ax_ylim = (-15.0, 15.0)
        self.ax.set_xlim(self.ax_xlim)
        self.ax.set_ylim(self.ax_ylim)
        self.ax.set_title("Obstacles-Space Arcs")

        # ====== 新增：自由空间窗口 ======
        self.fig_free, self.ax_free = plt.subplots(figsize=(7, 5))
        self.ax_free.set_aspect('equal')
        self.ax_free.set_xlim(self.ax_xlim)
        self.ax_free.set_ylim(self.ax_ylim)
        self.ax_free.set_title("Free-Space Arcs (No Obstacles)")

        self.fps = fps

        # —— 圆弧列表（动态增减）——
        self.arcs = []
        # —— 障碍 & 目标 ——
        self.circ_obs = []
        self.arcs_free = []
        self.circ_free = []
        self.target_circle = None
        self.target_circle_free = None
        self._is_open = True
        plt.show(block=False)

    # ---------- 每帧调用 ----------
    def update(self, arm, obstacles, target):
        if not self._is_open:
            return
        
        self.ax.clear()
        self.circ_obs.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.ax_xlim)
        self.ax.set_ylim(self.ax_ylim)
        self.ax.set_title("Obstacles-Space Arcs")

        # 2. 逐段画真实圆弧
        for seg in arm.segs_obs:
            self._draw_one_arc(self.ax, arm, seg)

        for x, y, r in obstacles:
            c = Circle((x, y), r, color='crimson', ec='k')
            self.ax.add_patch(c)
            self.circ_obs.append(c)

        # 4. 目标
        self.target_circle = Circle(target, 1.0, color='lime', ec='k')
        self.ax.add_patch(self.target_circle)
    
        # 5. 立即刷新
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # --- 新增：自由空间窗口绘制 ---
        self.ax_free.clear()
        self.circ_free.clear()
        self.ax_free.set_aspect('equal')
        self.ax_free.set_xlim(self.ax_xlim)
        self.ax_free.set_ylim(self.ax_ylim)
        self.ax_free.set_title("Free-Space Arcs (No Obstacles)")

        # 只画 segs_free，没有障碍物，也没有目标
        for seg in arm.segs_free:
            self._draw_one_arc(self.ax_free, arm, seg)

        self.target_circle_free = Circle(target, 1.0, color='lime', ec='k')
        self.ax_free.add_patch(self.target_circle_free)

        for x, y, r in obstacles:
            c = Circle((x, y), r,
                       facecolor=(0.9, 0.1, 0.2, 0.65),  # R,G,B,α  65 % 不透明
                       edgecolor='k')
            self.ax_free.add_patch(c)
            self.circ_free.append(c)

        # 刷新
        self.fig_free.canvas.draw_idle()
        self.fig_free.canvas.flush_events()

    # ---------- 画一段 PCC 圆弧（C¹ 连续）  ----------
    def _draw_one_arc(self, ax, arm, seg):
        """
        seg 必须有：p_start, p_end, L, kappa
        用 Matplotlib Arc 画真实圆弧（无近似）
        """
        p0 = seg.p_start
        p1 = seg.p_end
        k  = seg.kappa
        L  = seg.L     

        if L < 1e-3:
            return  # 太短不画

        if ax == self.ax:
            ax_name = "obstacle"
            idx = arm.segs_obs.index(seg)
        else:
            ax_name = "free"
            idx = arm.segs_free.index(seg)

        if abs(k) < 0.001:          # 直线 → 用细线
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', lw=3)
            # print(f"Drawing line in ", ax_name, ": p0 = ({float(p0[0]):.2f}, {float(p0[1]):.2f}), p1 = ({float(p1[0]):.2f}, {float(p1[1]):.2f}), kappa = {float(k):.2f}, L = {float(L):.2f}")
            return
        
        # 1. 段间切线连续：本段起始角 = 前段末端方向（世界系）
        if idx > 0:                           # 非首段
            world_tangent = seg.dir_start
        else:                                 # 首段用默认 x 轴
            world_tangent = np.array([1.0, 0.0])

        # 圆心、半径、角度
        r = 1.0 / abs(k)
        theta = L / r                        # 总圆心角
        angle_sign = np.sign(k)
        sweep = angle_sign * theta

        

        # 圆心坐标（2D 平面）
        mid = (p0 + p1) / 2.0
        chord_vec = p1 - p0
        perp = np.array([-chord_vec[1], chord_vec[0]])  # 垂直向量
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        # 圆心在垂直方向，距离 = r * cos(θ/2)
        dist_center = r * np.cos(theta / 2.0)
        center = mid + perp * angle_sign * dist_center

        # 4. 局部角度：世界切线 → 圆心局部系
        center_to_p0 = p0 - center                       # 局部 x 轴方向
        local_x = center_to_p0 / (np.linalg.norm(center_to_p0) + 1e-12)
        if local_x[1] == 0:
            start_angle = 0.0 if local_x[0] > 0 else np.pi
        else:
            start_angle = np.arctan2(local_x[1], local_x[0])
        # 5. 圆弧扫过角度
        sweep = angle_sign * theta

        # Matplotlib Arc：椭圆 → 圆（width=height=2r）
        if sweep >= 0:
            arc = Arc(center, 2 * r, 2 * r,
                      theta1=np.degrees(start_angle),
                      theta2=np.degrees(start_angle + sweep),
                      color='steelblue', lw=3)
            # print(f"Drawing arc in ", ax_name, ": p0 = ({float(p0[0]):.2f}, {float(p0[1]):.2f}), p1 = ({float(p1[0]):.2f}, {float(p1[1]):.2f}), "
            #     f"kappa = {float(k):.2f}, kappa_target = {float(seg.kappa_target):.2f}, L = {float(L):.2f}, r = {float(r):.2f}, theta = {float(theta) / np.pi * 180.0:.2f}, "
            #     f"start_angle = {np.degrees(float(start_angle)):.2f}, end_angle = {np.degrees(float(start_angle) + float(sweep)):.2f}")
        else:
            arc = Arc(center, 2 * r, 2 * r,
                        theta1=np.degrees(start_angle + sweep),
                        theta2=np.degrees(start_angle),
                        color='steelblue', lw=3)
            # print(f"Drawing arc in ", ax_name, ": p0 = ({float(p0[0]):.2f}, {float(p0[1]):.2f}), p1 = ({float(p1[0]):.2f}, {float(p1[1]):.2f}), "
            #     f"kappa = {float(k):.2f}, kappa_target = {float(seg.kappa_target):.2f}, L = {float(L):.2f}, r = {float(r):.2f}, theta = {float(theta) / np.pi * 180.0:.2f}, "
            #     f"start_angle = {np.degrees(float(start_angle) + float(sweep)):.2f}, end_angle = {np.degrees(float(start_angle)):.2f}")
        
        self.arcs.append(ax.add_patch(arc))

    def close(self):
        self._is_open = False
        plt.close(self.fig)