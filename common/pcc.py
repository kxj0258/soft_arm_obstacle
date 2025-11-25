import numpy as np
from typing import List

# ---------------- 2D ---------------- #
class PCCSeg2D:
    __slots__ = ("seg_id", "L", "kappa", "kappa_target", "kappa_fixed", "p_start", "p_end", 
                 "dir_start", "dir_end", "center",
                "l_left", "l_right",
                "width")

    def __init__(self, seg_id: int, L: float, kappa: float = None,
                 p_start: np.ndarray = None, dir_start: np.ndarray = None,
                 width: float = 1.0):
        self.seg_id = seg_id
        self.L = L
        self.width = width
        self.l_left = self.l_right = L         # 初始无弯曲，两侧长度=弧长
        self.drive(0.0, 0.0)              # 初始化 κ_target
        if kappa is None:
            self.kappa = self.kappa_target
        else:
            self.kappa = kappa
            self.kappa_target = kappa
            self.l_left = L * (1 - kappa * width / 2.0)
            self.l_right = L * (1 + kappa * width / 2.0)
        self.kappa_fixed = False           # 是否固定不动
        self.p_start = p_start if p_start is not None else np.zeros(2)
        self.p_end = None               # 将在 fk() 中自动计算
        self.dir_start: np.ndarray = dir_start if dir_start is not None else np.array([1.0, 0.0])   # 默认 x 轴
        self.dir_end: np.ndarray = None
        self.center: np.ndarray = None
        self.fk()

    def drive(self, dl_left: float, dl_right: float):
        """
        给定左右肌腱相对初始长度的变化量（正=拉长，负=缩短），
        根据几何关系更新 self.l_left、self.l_right 并计算 κ_target。
        注意：这里假设肌腱贴在圆弧最外侧，即距中性轴 ±width/2。
        """
        # 1. 更新长度
        self.l_left  += dl_left
        self.l_right += dl_right

        # 2. 中性层长度（平均长度）≈ 弧长 L
        L_neutral = 0.5 * (self.l_left + self.l_right)
        self.L = L_neutral          # 如果你想让总弧长随驱动变化，可保留；否则用原 L

        # 3. 曲率半径估算
        delta = self.l_right - self.l_left          # 长度差
        if abs(delta) < 1e-8:                       # 两侧相等 → 直线
            self.kappa_target = 0.0
            
        else:
            # 几何关系：delta = width * θ，而 θ = L_neutral / R
            # 得 R = width * L_neutral / delta
            # 故 κ = 1/R = delta / (width * L_neutral)
            self.kappa_target = delta / (self.width * L_neutral)

    # ---------- 工具：2D 旋转矩阵 ----------
    @staticmethod
    def _rot2d(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])


    def fk(self, T0: np.ndarray = None):
        """
        正向运动学：T0 是**世界系**起点位姿（3×3）。
        返回新的 3×3 齐次矩阵，并自动填写 self.p_end, self.dir_end。
        若 T0 为 None，则默认以 self.p_start 为起点。
        """
        if T0 is None:
            T0 = np.eye(3)
            T0[:2, 2] = self.p_start

        k = self.kappa

        if abs(k) < 0.001:                           # 直线
            T_local = np.array([[1, 0, self.L],
                                [0, 1, 0],
                                [0, 0, 1]])
        else:                                       # 圆弧
            r = 1.0 / k
            ang = self.L * k
            c, s = np.cos(ang), np.sin(ang)
            T_local = np.array([[c, -s, r * s],
                                [s,  c, r * (1 - c)],
                                [0,  0, 1]])
        T_world = T0 @ T_local
        # 自动填写终点
        self.p_end = T_world[:2, 2].copy()
        self.dir_end = T_world[:2, 0]          # ✅ 世界系切线
        if abs(k) >= 0.001:
            # 圆心坐标（2D 平面）
            r = 1.0 / k
            mid = (self.p_start + self.p_end) / 2.0
            chord_vec = self.p_end - self.p_start
            perp = np.array([-chord_vec[1], chord_vec[0]])  # 垂直向量
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            # 圆心在垂直方向，距离 = r * cos(θ/2)
            dist_center = r * np.cos(self.L * k / 2.0)
            self.center = mid + perp * np.sign(k) * dist_center
        else:
            self.center = None

        return T_world


class ContinuumArm2D:
    def __init__(self, segs: List[PCCSeg2D]):
        self.segs = segs

    def fk(self, T0: np.ndarray = None):
        """链式 FK，并自动把前一段 p_end 设为下一段 p_start"""
        if T0 is None:
            T = np.eye(3)
            T[:2, 2] = self.segs[0].p_start
        else:
            T = T0
        for i, s in enumerate(self.segs):
            if i > 0:                           # 非首段
                s.p_start = self.segs[i-1].p_end
                s.dir_start = self.segs[i-1].dir_end
            T = s.fk(T)
        return T


# ---------------- 3D ---------------- #
class PCCSeg3D:
    __slots__ = ("seg_id", "L",
                "kappa1", "kappa2", "kappa1_target", "kappa2_target", 
                "kappa1_fixed", "kappa2_fixed",
                "p_start", "p_end", "dir_start", "dir_end",
                "l_up", "l_down", "l_left", "l_right",
                "width", "height")

    def __init__(self, seg_id: int, L: float,
                 kappa1: float = 0.0, kappa2: float = 0.0,
                 p_start: np.ndarray = None,
                 width: float = 0.02,
                 height: float = 0.02):
        self.seg_id = seg_id
        self.L = L
        self.width = width
        self.height = height
        # 初始无弯曲，四腱长度 = L
        self.l_up = self.l_down = self.l_left = self.l_right = L
        # 驱动计算目标曲率
        self.drive(0.0, 0.0, 0.0, 0.0)
        # 当前曲率
        self.kappa1 = kappa1 if kappa1 is not None else self.kappa1_target
        self.kappa2 = kappa2 if kappa2 is not None else self.kappa2_target
        self.kappa1_fixed = False
        self.kappa2_fixed = False
        # 起点
        self.p_start = p_start if p_start is not None else np.zeros(3)
        self.p_end = None
        self.dir_start = np.array([1.0, 0.0, 0.0])
        self.dir_end = None

    # ---------- 驱动接口 ----------
    def drive(self, dl_up: float, dl_down: float,
              dl_left: float, dl_right: float):
        """
        根据四腱长度变化量更新 l_xxx 并计算 (kappa1_target, kappa2_target)
        约定：
          κ1 对应绕 Y 轴（上下弯曲，由 up/down 控制）
          κ2 对应绕 Z 轴（左右弯曲，由 left/right 控制）
        """
        self.l_up += dl_up
        self.l_down += dl_down
        self.l_left += dl_left
        self.l_right += dl_right

        # 中性层长度
        L_neutral = 0.25 * (self.l_up + self.l_down + self.l_left + self.l_right)
        self.L = L_neutral

        # 上下差 → 绕 Y 轴曲率
        delta_ud = self.l_up - self.l_down
        # 左右差 → 绕 Z 轴曲率
        delta_lr = self.l_left - self.l_right

        # 小变形公式：θ = Δl / d，κ = θ / L_neutral
        if abs(delta_ud) < 1e-8:
            self.kappa1_target = 0.0
        else:
            self.kappa1_target = delta_ud / (self.height * L_neutral)

        if abs(delta_lr) < 1e-8:
            self.kappa2_target = 0.0
        else:
            self.kappa2_target = delta_lr / (self.width * L_neutral)


    def fk(self, T0: np.ndarray = None):
        if T0 is None:
            T0 = np.eye(4)
            T0[:3, 3] = self.p_start

        k1, k2, L = self.kappa1, self.kappa2, self.L
        k = np.hypot(k1, k2)
        if k < 1e-6:                                    # 直线
            T_local = np.eye(4)
            T_local[0, 3] = L
        else:                                           # 圆弧
            r = 1.0 / k
            ang = L * k
            axis = np.array([k2, 0, k1]) / k
            from scipy.spatial.transform import Rotation as R
            R_mat = R.from_rotvec(axis * ang).as_matrix()
            p = (np.eye(3) - R_mat) @ (r * axis)
            T_local = np.eye(4)
            T_local[:3, :3], T_local[:3, 3] = R_mat, p
        T_world = T0 @ T_local
        self.p_end = T_world[:3, 3].copy()
        self.dir_end = T_world[:3, 0]
        return T_world


class ContinuumArm3D:
    def __init__(self, segs: List[PCCSeg3D]):
        self.segs = segs

    def fk(self, T0: np.ndarray = None):
        if T0 is None:
            T0 = np.eye(4)
            T0[:3, 3] = self.segs[0].p_start
        else:
            T = T0
        for i, s in enumerate(self.segs):
            if i > 0:
                s.p_start = self.segs[i-1].p_end
                s.dir_start = self.segs[i-1].dir_end
            T = s.fk(T)
        return T