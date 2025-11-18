from common.pcc import ContinuumArm2D, PCCSeg2D
import numpy as np
from typing import List, Optional, Tuple

class ArmModel2D(ContinuumArm2D):
    """
    基于 PCCSeg2D 的 2D 软体臂模型，支持碰撞插入新节点。
    1. self.n_seg  → 直接表示 **常曲率圆弧段数**（len(self.segs)）
    2. 基座原点 = 第 0 段 p_start，可由外部推进自由度修改 x∈[0, 0.50]
    3. insert_segment_at 接收「碰撞点坐标 (x_col, y_col)」，
       按「段内比例」把该段拆成两段，前段终点＝碰撞点，后段保持原 κ。
    双链维护：
    1. segs_free → 自由空间 PCC（仅曲率驱动）
    2. segs_obs  → 在 free 基础上做碰撞检测 & 插入，得到「真实世界」臂
    外部只读 segs_obs（渲染、物理、FK 都用它）
    """
    # ========= 1. 初始化 ========= #
    def __init__(self, segs=None, base_x=0.0, obstacles: List[Tuple[float, float, float]] = None):
        """
        # 障碍物列表: [(cx, cy, r), ...]
        只在 obs 链上做碰撞检测
        只在 free 链上做动作驱动
        由外部控制基座 x 位置
        由外部控制各段曲率 κ
        由内部做碰撞检测 & 拆段
        """
        self.obstacles = obstacles if obstacles is not None else []

        # 推进自由度（只改 free 和 obs 链起点 x）
        if base_x < 0.0 or base_x > 50.0:
            raise ValueError("base_x must be in [0.0, 50.0]")
        self.base_x = base_x

        # ① 自由链（初始状态）
        self.segs_free = (segs or [PCCSeg2D(L=10.0, kappa=0.0, p_start=np.array([self.base_x, 0.0])),
                                   PCCSeg2D(L=15.0, kappa=0.0)])
        self.n_free = len(self.segs_free)
        self.T_free = self.fk_free()               # 自动计算起点/终点

        # ② 观测链 = 自由链深拷贝（第一次无碰撞）
        self.segs_obs = [_deep_copy_seg(s) for s in self.segs_free]
        self.n_obs  = len(self.segs_obs)
        self._update_obs_from_free()  # 同步 obs 链
        self.T_obs = self.fk_obs()
    
    # ========= 2. 动作：只驱动 free 链 ========= #
    def apply_action(self, dx: float, dkappa: np.ndarray):
        """基座 x∈[0, 50.0] cm，只改 free 链起点"""
        self.base_x = self.base_x + dx
        self.base_x = np.clip(self.base_x, 0.0, 50.0)
        self.segs_free[0].p_start[0] = self.base_x
        """
        dkappa: [Δκ1, Δκ2, ...] 与 free 段数相同
        只改 free 链，再同步到 obs 链
        """
        for i, dk in enumerate(dkappa):
            if i < self.n_free:
                self.segs_free[i].kappa = np.clip(self.segs_free[i].kappa + dk, -3, 3)

        self.T_free = self.fk_free()               # 重新 FK

        self._update_obs_from_free()

    def _update_obs_from_free(self):
        """同步 obs 链到 free 链（起点、曲率）"""
        for i in range(self.n_obs):
            self.segs_obs[i].p_start = self.segs_free[i].p_start.copy()
            self.segs_obs[i].kappa  = self.segs_free[i].kappa
        self.fk_obs()

        # 更新 obs 链后，做碰撞检测 & 解析
        self.resolve_impulse()

        self.T_obs = self.fk_obs()

    def resolve_impulse(self):
        """对每段圆弧做弹簧-阻尼冲量，并拆段"""
        segs = self.segs_obs  # 只检测 obs 链   
        for cx, cy, r in self.obstacles:
            for i, seg in enumerate(segs):
                info = seg_circle_nearest(seg, np.array([cx, cy]), r)
                if info:
                    p, n, depth, col_point, frac = info   # 多收一个碰撞点
                    # 把圆上碰撞点一并传给 arm
                    self.insert_segment_at(i, frac, n, col_point)
                    break        # 同一障碍只处理一次



    # ========= 3. 碰撞拆段 ========= #
    def insert_segment_at(self,
                          i: int,
                          frac: float,
                          normal: np.ndarray,
                          col_point: Tuple[float, float]):
        """
        i        : 第 i 段常曲率圆弧（0-based）
        frac     : 在该段内的碰撞比例 0~1
        normal   : 碰撞法向（单位 2D 向量），备用
        col_point: (x_col, y_col) 碰撞点坐标（世界系）
        """
        if not (0 <= i < len(self.segs_obs)):
            return
        seg = self.segs_obs[i]
        L_orig = seg.L
        k_orig = seg.kappa

        # ---- 3.1 拆长度 ----
        L1 = L_orig * frac
        L2 = L_orig * (1.0 - frac)

        # ---- 3.2 前段：终点 = 碰撞点 ----
        p_start = seg.p_start
        p_col   = np.asarray(col_point, dtype=float)
        k_new_front = fit_kappa(p_start, p_col, L1)
        # 符号：保持与原来同向（可再细调）
        k_new_front = np.copysign(k_new_front, k_orig)
        seg_front = PCCSeg2D(L1, k_new_front)

        # ---- 3.3 后段：保持原 κ ----
        seg_back = PCCSeg2D(L2, k_orig)

        # ---- 3.4 替换列表 ----
        self.segs_obs[i:i + 1] = [seg_front, seg_back]
        self.n_seg_obs = len(self.segs_obs)          # 更新段数
        self.fk_obs()                            # 重新 FK

    # ========= 4. 辅助：整条臂 FK 并自动接力起点 ========= #
    def fk_free(self, T0: np.ndarray = None):
        """链式 FK，并自动把前一段 p_end 设为下一段 p_start"""
        if T0 is None:
            T = np.eye(3)
            T[:2, 2] = self.segs_free[0].p_start
        else:
            T = T0

        for i, s in enumerate(self.segs_free):            
            if i > 0:
                s.p_start = self.segs_free[i-1].p_end
                s.dir_start = self.segs_free[i-1].dir_end     
            T = s.fk(T)       
        return T
    
    def fk_obs(self, T0: np.ndarray = None):
        """链式 FK，并自动把前一段 p_end 设为下一段 p_start"""
        if T0 is None:
            T = np.eye(3)
            T[:2, 2] = self.segs_obs[0].p_start
        else:
            T = T0
        for i, s in enumerate(self.segs_obs):            
            if i > 0:
                s.p_start = self.segs_obs[i-1].p_end     
            T = s.fk(T)       
        return T

# 三点已知：起点、终点、弧长 → 求曲率 κ
def fit_kappa(p0: np.ndarray, p1: np.ndarray, L: float) -> float:
    """
    平面常曲率圆弧：已知弦长 d 和弧长 L，反求曲率 κ。
    几何关系：d = 2 r sin(θ/2) ，L = r θ ，其中 r = 1/|κ|
    → 解非线性方程 2 sin(δ κ)/κ = d ，δ = L/2
    这里用一次 Newton 迭代即可收敛。
    """
    d = np.linalg.norm(p1 - p0)
    if d > L:                       # 不可能用圆弧连接
        return 0.0                  # 退化为直线
    delta = L * 0.5
    k = 0.0                         # 初值
    for _ in range(5):              # 5 次 Newton 足够
        f  = 2.0 * np.sin(delta * k) / (k + 1e-12) - d
        df = 2.0 * (delta * np.cos(delta * k) * k - np.sin(delta * k)) / (k * k + 1e-12)
        k -= f / (df + 1e-12)
    return k

def _deep_copy_seg(seg: PCCSeg2D) -> PCCSeg2D:
    """深拷贝一段 PCC（含起点坐标）"""
    new = PCCSeg2D(L=seg.L, kappa=seg.kappa, p_start=seg.p_start.copy())
    return new


def seg_circle_nearest(seg: 'PCCSeg2D',
                       c: np.ndarray,
                       r: float
                       ) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    圆弧-圆 最近点检测（2D）
    输入:
        seg : PCCSeg2D 对象，要求已填写 p_start / p_end
        c   : 障碍圆心 (2,)
        r   : 障碍半径
    返回:
        None
        (nearest, normal, depth, col_point)
            nearest  : 圆弧上距圆心最近点（世界坐标）
            normal   : nearest → c 的单位向量
            depth    : 穿透深度 (>0 表示相交)
            col_point: 障碍圆上真正的碰撞点 = nearest + normal*depth
            frac     : 碰撞点在圆弧段内的比例 0~1
    """
    # 1. 圆弧端点
    p_start = seg.p_start
    p_end   = seg.p_end
    if p_start is None or p_end is None:
        return None

    # 2. 圆弧参数：总弦长、总扫角、半径、圆心
    chord = p_end - p_start
    chord_len = np.linalg.norm(chord)
    k = seg.kappa
    if abs(k) < 0.1:                      # 退化直线，用旧线段逻辑
        return _line_circle_nearest(p_start, p_end, c, r)

    total_ang = seg.L * k
    R_arc = 1.0 / k
    # 圆弧局部圆心（世界系）
    # 局部坐标：起点 (0,0)，终点 (chord,0)，圆弧在 +y 侧
    normal_dir = np.array([-chord[1], chord[0]])    # 90° 旋转
    normal_dir /= np.linalg.norm(normal_dir) + 1e-12
    # 局部圆心 → 世界圆心
    if k > 0:
        c_arc_local = normal_dir * R_arc
    else:
        c_arc_local = -normal_dir * R_arc
    c_arc_world = p_start + c_arc_local

    # 3. 最近点参数：把障碍圆心投影到圆弧所在圆
    vec = c - c_arc_world
    d_centers = np.linalg.norm(vec)
    if d_centers > abs(R_arc) + r:       # 两圆相离
        return None
    # 投影方向
    if d_centers < 1e-12:                # 两圆心重合
        t = 0.0
    else:
        t = np.arctan2(vec[1], vec[0]) - np.arctan2(chord[1], chord[0])
        # 归到 [-π, π]
        t = np.arctan2(np.sin(t), np.cos(t))

    # 4. 截断到圆弧实际扫角范围
    t_min = 0.0
    t_max = total_ang
    t = np.clip(t, min(t_min, t_max), max(t_min, t_max))

    # 5. 圆弧上最近点（世界坐标）
    ang = t
    c_local, s_local = np.cos(ang), np.sin(ang)
    if k > 0:
        nearest_local = np.array([R_arc * np.sin(ang),
                                  R_arc * (1 - np.cos(ang))])
    else:
        nearest_local = np.array([R_arc * np.sin(ang),
                                  -R_arc * (1 - np.cos(ang))])
    # 局部 → 世界
    rot = np.array([[ chord[0], -chord[1]],
                    [ chord[1],  chord[0]]]) / chord_len
    nearest = p_start + rot @ nearest_local

    # 6. 穿透检测
    dist_vec = c - nearest
    dist = np.linalg.norm(dist_vec)
    if dist >= r or dist < 1e-12:
        return None
    normal = dist_vec / dist
    depth = r - dist
    col_point = nearest - normal * depth
    # ✅ 把角度 t 归一化为 0~1 的段内比例
    frac = t / total_ang if abs(total_ang) > 1e-6 else 0.0
    return nearest, normal, depth, col_point, frac


def _line_circle_nearest(p1, p2, c, r):
    """旧线段逻辑，备用"""
    seg = p2 - p1
    seg_len = np.linalg.norm(seg)
    if seg_len == 0:
        return None
    t = np.clip(np.dot(c - p1, seg) / seg_len**2, 0, 1)
    nearest = p1 + t * seg
    dist_vec = c - nearest
    dist = np.linalg.norm(dist_vec)
    if dist < r and dist > 1e-6:
        normal = dist_vec / dist
        depth = r - dist
        col_point = nearest + normal * depth
        return nearest, normal, depth, col_point, t
    return None