import numpy as np
from typing import List

# ---------------- 2D ---------------- #
class PCCSeg2D:
    __slots__ = ("seg_id", "L", "kappa", "kappa_target", "kappa_fixed", "p_start", "p_end", 
                 "dir_start", "dir_end", "center")

    def __init__(self, seg_id: int, L: float, kappa: float,
                 p_start: np.ndarray = None, dir_start: np.ndarray = None):
        self.seg_id = seg_id
        self.L, self.kappa = L, kappa
        self.kappa_target = kappa    # 目标曲率（用于平滑控制）
        self.kappa_fixed = False           # 是否固定不动
        self.p_start = p_start if p_start is not None else np.zeros(2)
        self.p_end = None               # 将在 fk() 中自动计算
        self.dir_start: np.ndarray = dir_start if dir_start is not None else np.array([1.0, 0.0])   # 默认 x 轴
        self.dir_end: np.ndarray = None
        self.center: np.ndarray = None
        self.fk()

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
    __slots__ = ("L", "kappa1", "kappa2", "p_start", "p_end")

    def __init__(self, L: float,
                 kappa1: float, kappa2: float,
                 p_start: np.ndarray = None):
        self.L, self.kappa1, self.kappa2 = L, kappa1, kappa2
        self.p_start = p_start if p_start is not None else np.zeros(3)
        self.p_end = None

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

    def fk(self):
        T = np.eye(4)
        for i, s in enumerate(self.segs):
            if i > 0:
                s.p_start = self.segs[i-1].p_end
                s.dir_start = self.segs[i-1].dir_end
            T = s.fk(T)
        return T