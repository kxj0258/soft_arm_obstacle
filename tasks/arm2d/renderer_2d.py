import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle


class Renderer2D:
    def __init__(self, fps=30):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-20.0, 220.0)
        self.ax.set_ylim(-60.0, 60.0)
        self.fps = fps

        # —— 圆弧列表（动态增减）——
        self.arcs = []
        # —— 障碍 & 目标 ——
        self.circ_obs = []
        self.target_circle = None
        self._is_open = True
        plt.show(block=False)

    # ---------- 每帧调用 ----------
    def update(self, arm, obstacles, target):
        if not self._is_open:
            return
        
        self.ax.clear() 
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10.0, 80.0)
        self.ax.set_ylim(-30.0, 30.0)

        # 2. 逐段画真实圆弧
        for seg in arm.segs_obs:
            self._draw_one_arc(arm, seg)

        for x, y, r in obstacles:
            c = Circle((x, y), r, color='crimson', ec='k')
            self.ax.add_patch(c)
            self.circ_obs.append(c)

        # 4. 目标
        self.target_circle = Circle(target, 2.0, color='lime', ec='k')
        self.ax.add_patch(self.target_circle)
    
        # 5. 立即刷新
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ---------- 画一段 PCC 圆弧（C¹ 连续）  ----------
    def _draw_one_arc(self, arm, seg):
        """
        seg 必须有：p_start, p_end, L, kappa
        用 Matplotlib Arc 画真实圆弧（无近似）
        """
        p0 = seg.p_start
        p1 = seg.p_end
        k  = seg.kappa
        L  = seg.L        

        if abs(k) < 0.01:          # 直线 → 用细线
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', lw=1)
            print("Drawing line: p0 =", p0, "p1 =", p1, "kappa =", k, "L =", L)
            return
        
        # 1. 段间切线连续：本段起始角 = 前段末端方向（世界系）
        idx = arm.segs_obs.index(seg)
        if idx > 0:                           # 非首段
            world_tangent = seg.dir_start
        else:                                 # 首段用默认 x 轴
            world_tangent = np.array([1.0, 0.0])

        # 圆心、半径、角度
        r = 1.0 / abs(k)
        theta = L / r                        # 总圆心角
        angle_sign = np.sign(k)
        sweep = angle_sign * theta

        print("Drawing arc: p0 =", p0, "p1 =", p1, "kappa =", k, "L =", L, "r =", r, "theta =", theta / np.pi * 180.0)

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
        arc = Arc(center, 2 * r, 2 * r,
                  theta1=np.degrees(start_angle),
                  theta2=np.degrees(start_angle + sweep),
                  color='steelblue', lw=1)
        
        self.arcs.append(self.ax.add_patch(arc))

    def close(self):
        self._is_open = False
        plt.close(self.fig)