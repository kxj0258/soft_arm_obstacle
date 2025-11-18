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
                print("Segment", i, "kappa before:", self.segs_free[i].kappa)
                self.segs_free[i].kappa = np.clip(self.segs_free[i].kappa + dk, -1, 1)
                print("Segment", i, "kappa after:", self.segs_free[i].kappa)

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
        seg_front = PCCSeg2D(L1, k_new_front, p_start, seg.dir_start)

        print("seg_front: L =", L1, "kappa =", k_new_front, "p_start =", p_start, "dir_start =", seg.dir_start)

        # ---- 3.3 后段：保持原 κ ----
        seg_back = PCCSeg2D(L2, k_orig, p_col, seg_front.dir_end)

        print("seg_back: L =", L2, "kappa =", k_orig, "p_start =", p_col, "dir_start =", seg_front.dir_end)
        

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
        
    """

    # 1. 基本检查
    p_start = seg.p_start
    p_end = seg.p_end
    if p_start is None or p_end is None:
        return None

    k = seg.kappa
    # 退化为直线段或未计算 center 的情况，回退到线段逻辑
    if seg.center is None or abs(k) < 1e-6:
        return _line_circle_nearest(p_start, p_end, c, r)

    c_arc = seg.center
    R = abs(1.0 / k)
    eps = 1e-9

    # 辅助：角度归一化与区间重叠判断
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _intervals_overlap(start1, end1, start2, end2):
        # 将两个区间都映射到实数轴（end = start + length），尝试平移 2pi
        if end1 < start1:
            end1 += 2 * np.pi
        if end2 < start2:
            end2 += 2 * np.pi
        # try shifts of second interval by -2pi, 0, +2pi
        for shift in (-2 * np.pi, 0.0, 2 * np.pi):
            s2 = start2 + shift
            e2 = end2 + shift
            if max(start1, s2) <= min(end1, e2) + 1e-12:
                return True
        return False

    # 2. 先做端点检查（端点在障碍圆内即为碰撞）
    d_start = np.linalg.norm(p_start - c)
    d_end = np.linalg.norm(p_end - c)
    if d_start <= r + eps or d_end <= r + eps:
        # 优先选离障碍心最近的端点
        if d_start <= d_end:
            nearest = p_start.copy()
            t_rel = 0.0
        else:
            nearest = p_end.copy()
            t_rel = seg.L
        dist = min(d_start, d_end)
        if dist < 1e-12:
            return None
        normal = (c - nearest) / (dist + 1e-12)
        depth = r - dist
        col_point = nearest - normal * depth
        # frac：端点比例
        frac = 0.0 if d_start <= d_end else 1.0
        return nearest, normal, depth, col_point, frac

    # 3. 两圆几何关系
    vec = c - c_arc
    d = np.linalg.norm(vec)

    # 快速排除：两圆没有交集且弧的端点也不在圆内
    if d > R + r + eps:
        return None

    # 中心重合处理
    if d < eps:
        # 若障碍圆覆盖整圆（r >= R），则任意弧点都处于障碍内，选起点
        if r >= R - eps:
            nearest = p_start.copy()
            dist = np.linalg.norm(c - nearest)
            if dist < 1e-12:
                return None
            normal = (c - nearest) / (dist + 1e-12)
            depth = r - dist
            col_point = nearest - normal * depth
            frac = 0.0
            return nearest, normal, depth, col_point, frac
        else:
            # 障碍圆在弧圆内部，不一定与弧相交（端点已检查），排除
            return None

    # 4. 计算整圆与障碍圆的交角区间（若有交点）
    # cos_delta = (d^2 + R^2 - r^2) / (2*d*R)
    cos_delta = (d * d + R * R - r * r) / (2.0 * d * R)
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    delta = np.arccos(cos_delta)
    alpha0 = np.arctan2(vec[1], vec[0])
    cover_start = _wrap(alpha0 - delta)
    cover_end = _wrap(alpha0 + delta)

    # 5. 弧角区间
    ang_start = np.arctan2(p_start[1] - c_arc[1], p_start[0] - c_arc[0])
    total_ang = seg.L * k
    ang_end = ang_start + total_ang

    # 判断整圆被覆盖情形（若 r >= d + R => 障碍圆覆盖整圆）
    if r >= d + R - eps:
        cover_start_full = ang_start - 2 * np.pi
        cover_end_full = ang_start + 2 * np.pi
        overlap = _intervals_overlap(ang_start, ang_end, cover_start_full, cover_end_full)
    else:
        overlap = _intervals_overlap(ang_start, ang_end, cover_start, cover_end)

    # 如果没有角区间重合，则没有在弧上的交集（端点已检查过）
    if not overlap:
        return None

    # 6. 若有重合，计算弧上最近点：沿向量 alpha0（圆心指向障碍心方向）的投影并裁剪到弧区间
    alpha_proj = alpha0
    # 将投影相对于起点的相对角度
    t = _wrap(alpha_proj - ang_start)
    t_clamped = np.clip(t, min(0.0, total_ang), max(0.0, total_ang))
    ang_nearest = ang_start + t_clamped
    nearest = c_arc + np.array([np.cos(ang_nearest), np.sin(ang_nearest)]) * R

    # 7. 穿透检测与返回
    dist_vec = c - nearest
    dist = np.linalg.norm(dist_vec)
    if dist < 1e-12:
        return None
    if dist >= r + eps:
        return None
    normal = dist_vec / (dist + 1e-12)
    depth = r - dist
    col_point = nearest - normal * depth
    frac = (t_clamped / total_ang) if abs(total_ang) > 1e-12 else 0.0
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