from common.pcc import ContinuumArm2D, PCCSeg2D
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CollisionPoint:
    """记录一次碰撞的信息
    """
    pos: np.ndarray         # 碰撞发生的位置（通常为障碍圆上的碰撞点）
    obstacle: Tuple[float, float, float]  # 障碍物 (cx, cy, r)
    seg_id: int          # 发生碰撞的圆弧段的索引
    nearest: np.ndarray    # 圆弧上最近点（世界坐标）
    normal: np.ndarray     # 碰撞法向量（单位）
    depth: float           # 穿透深度
    col_point: np.ndarray  # 障碍圆上的真正碰撞点
    frac: float            # 碰撞点在该段内的比例 0~1


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

        # 碰撞点列表（每个元素为 CollisionPoint）
        self.collision_points: List['CollisionPoint'] = []
        # 碰撞点数量
        self.n_collision_points: int = 0

        # 推进自由度（只改 free 和 obs 链起点 x）
        if base_x < 0.0 or base_x > 50.0:
            raise ValueError("base_x must be in [0.0, 50.0]")
        self.base_x = base_x

        # ① 自由链（初始状态）
        self.segs_free = (segs or [PCCSeg2D(seg_id = 0, L=10.0, kappa=0.0, p_start=np.array([self.base_x, 0.0])),
                                   PCCSeg2D(seg_id = 1, L=15.0, kappa=0.0)])
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
        self.segs_obs[0].p_start[0] = self.base_x
        """
        dkappa: [Δκ1, Δκ2, ...] 与 free 段数相同
        只改 free 链，再同步到 obs 链
        """
        for i, dk in enumerate(dkappa):
            if i < self.n_free:
                # print("Segment", i, "kappa before:", self.segs_free[i].kappa)
                self.segs_free[i].kappa = np.clip(self.segs_free[i].kappa + dk, -1, 1)
                self.segs_free[i].kappa_target = self.segs_free[i].kappa

                # for seg in self.segs_obs:
                #     if seg.seg_id == self.segs_free[i].seg_id:
                #         if seg.kappa_fixed:
                #             seg.kappa_target = self.segs_free[i].kappa
                #         else:
                #             seg.kappa = self.segs_free[i].kappa
                #             seg.kappa_target = self.segs_free[i].kappa
                # print("Segment", i, "kappa after:", self.segs_free[i].kappa)

        self.T_free = self.fk_free()

        self.resolve_impulse("free") 

        self._update_obs_from_free()

    def _update_obs_from_free(self):
        # 同步 free 到 obs 链
        for seg_free in self.segs_free:
            for seg_obs in self.segs_obs:
                if seg_obs.seg_id == seg_free.seg_id:
                    seg_obs.kappa = seg_free.kappa
                    seg_obs.kappa_target = seg_free.kappa
        # 检查碰撞点列表
        free_collision_ids = []
        for i, collision_point in enumerate(self.collision_points):
            isFreeCollision = self.insert_segment_at(collision_point.seg_id, collision_point.col_point, collision_point.obstacle, collision_point.frac)
            if isFreeCollision:
                free_collision_ids.append(i)
        
        for idx in free_collision_ids:
            self.collision_points.pop(idx)
        self.n_collision_points = len(self.collision_points)
        self.fk_obs()

        # 更新 obs 链后，做碰撞检测 & 解析
        # self.resolve_impulse("obs")

        self.T_obs = self.fk_obs()

    def resolve_impulse(self, mode: str = "free"):
        if mode == "free":
            segs = self.segs_free
        else:
            segs = self.segs_obs
        # # 清空上一次的碰撞记录
        # self.collision_points.clear()
        # self.n_collision_points = 0
        """对每段圆弧做弹簧-阻尼冲量，并拆段"""
        is_collision = False
        for cx, cy, r in self.obstacles:
            for i, seg in enumerate(segs):
                info = seg_circle_nearest(seg, np.array([cx, cy]), r)
                if info:
                    is_collision = True
                    p, n, depth, col_point, frac = info   # 多收一个碰撞点
                    is_new_collision = True
                    for cp in self.collision_points:
                        if cp.obstacle == (float(cx), float(cy), float(r)) and cp.seg_id == seg.seg_id:
                            # 已处理过该障碍物与该段
                            if cp.frac != frac:
                                cp.frac = frac
                            is_new_collision = False
                            break
                    if not is_new_collision:
                        continue
                    # 记录碰撞信息
                    cp = CollisionPoint(
                        pos=np.asarray(col_point, dtype=float),
                        obstacle=(float(cx), float(cy), float(r)),
                        seg_id=seg.seg_id,
                        nearest=np.asarray(p, dtype=float),
                        normal=np.asarray(n, dtype=float),
                        depth=float(depth),
                        col_point=np.asarray(col_point, dtype=float),
                        frac=float(frac)
                    )
                    self.collision_points.append(cp)
                    self.n_collision_points = len(self.collision_points)
                    print(
                        f"Collision detected: seg {int(seg.seg_id)}, "
                        f"circle center=({float(cx):.2f}, {float(cy):.2f}, r={float(r):.2f}), "
                        f"col_point=({float(col_point[0]):.2f}, {float(col_point[1]):.2f}), "
                        f"frac={float(frac):.2f}, "
                        f"depth={float(depth):.2f}, "
                        f"normal=({float(n[0]):.2f}, {float(n[1]):.2f}), "
                        f"p=({float(p[0]):.2f}, {float(p[1]):.2f})"
                    )
                    # 把圆上碰撞点一并传给 arm
                    # self.insert_segment_at(i, frac, n, col_point)
                    # break        # 同一障碍只处理一次
        
        # if not is_collision:
        #     # 无碰撞时，清空碰撞点列表
        #     if self.collision_points:
        #         self.collision_points.clear()
        #         self.n_collision_points = 0
        #     # 重置所有段的 kappa_fixed 状态
        #     for seg in self.segs_obs:
        #         seg.kappa_fixed = False


    # ========= 3. 碰撞拆段 ========= #
    def insert_segment_at(self,
                          i: int,
                          col_point: Tuple[float, float],
                          obstacle: Tuple[float, float, float],
                          frac: float):
        """
        i        : 第 i 段常曲率圆弧（0-based）
        col_point: (x_col, y_col) 碰撞点坐标（世界系）
        obstacle : (cx, cy, r) 障碍物参数
        frac     : 碰撞点在该段内的比例 0~1
        """
        if not (0 <= i < len(self.segs_obs)) or frac <= 0.0 or frac >= 1.0:
            return True
        seg = self.segs_obs[i]
        L_orig = seg.L
        k_orig = seg.kappa

        # ---- 3.2 前段：终点 = 碰撞点 ----
        p_start = seg.p_start
        dir_start = seg.dir_start
        p_col   = np.asarray(col_point, dtype=float)
        k_front, s_front = signed_curvature(p_start, dir_start, p_col)


        # 判断是否解除碰撞
        frac = s_front / L_orig
        free_frac = point_at_frac(p_start, dir_start, k_orig, frac, L_orig)        
        col2free= free_frac - p_col
        col2obs = obstacle[:2] - p_col
        print(f"col2free = ({float(col2free[0]):.2f}, {float(col2free[1]):.2f}), "
              f"col2obs = ({float(col2obs[0]):.2f}, {float(col2obs[1]):.2f})")
        print(f"Dot product = {float(np.dot(col2free, col2obs)):.2f}")
        if np.dot(col2free, col2obs) < 0:
            return True
        
        # ---- 3.3 新建两段 ----
        # k_front = np.copysign(k_front, k_orig)
        seg_front = PCCSeg2D(seg.seg_id, s_front, k_front, p_start, seg.dir_start)
        if abs(k_orig) > abs(1.0 / obstacle[2]):
            k_back = 1.0 / obstacle[2]
            k_back = np.copysign(k_back, k_orig)
        else:
            k_back = k_orig
        seg_back = PCCSeg2D(seg.seg_id, L_orig - s_front, k_back, p_col, seg_front.dir_end)
        # print(f"seg_front: L = {float(L1):.2f}, kappa = {float(k_new_front):.2f}, "
        #     f"p_start = ({float(p_start[0]):.2f}, {float(p_start[1]):.2f}), "
        #     f"dir_start = ({float(seg.dir_start[0]):.2f}, {float(seg.dir_start[1]):.2f})")

        # print(f"seg_back:  L = {float(L2):.2f}, kappa = {float(k_orig):.2f}, "
        #     f"p_start = ({float(p_col[0]):.2f}, {float(p_col[1]):.2f}), "
        #     f"dir_start = ({float(seg_front.dir_end[0]):.2f}, {float(seg_front.dir_end[1]):.2f})")
        
        # ---- 3.4 替换列表 ----
        self.segs_obs[i:i + 1] = [seg_front, seg_back]
        self.n_seg_obs = len(self.segs_obs)          # 更新段数
        self.fk_obs()                            # 重新 FK
        return False

    def merge_segments_at(self, i: int) -> bool:
        """
        把 self.segs_obs[i] 与 self.segs_obs[i+1] 合并成一条。
        成功返回 True，否则 False。
        """
        if i + 1 >= len(self.segs_obs):
            return False

        seg1 = self.segs_obs[i]
        seg2 = self.segs_obs[i + 1]

        # # 1. 合并条件：曲率同号且绝对值足够接近，方向连续
        # if not (abs(seg1.kappa - seg2.kappa) < 1e-4 and
        #         np.allclose(seg1.dir_end, seg2.dir_start, atol=1e-4)):
        #     return False

        # 2. 新长度
        new_L = seg1.L + seg2.L

        # 3. 新段：继承前段起点、后段终点，曲率不变
        new_seg = PCCSeg2D(
            seg_id=seg1.seg_id,      # 沿用原 ID
            L=new_L,
            kappa=seg1.kappa,
            p_start=seg1.p_start,
            dir_start=seg1.dir_start
        )
        new_seg.kappa_fixed = False

        # 4. 列表替换
        self.segs_obs[i:i + 2] = [new_seg]
        self.n_seg_obs = len(self.segs_obs)

        # # 5. 重新正运动学
        # self.fk_obs()
        return True


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
        for i, s1 in enumerate(self.segs_obs):
            if i < len(self.segs_obs) - 1:
                s2 = self.segs_obs[i + 1]
                if s1.seg_id == s2.seg_id and not s1.kappa_fixed and not s2.kappa_fixed:
                    self.merge_segments_at(i)        

        for i, s in enumerate(self.segs_obs):            
            if i > 0:
                s.p_start = self.segs_obs[i-1].p_end  
                s.dir_start = self.segs_obs[i-1].dir_end   
            T = s.fk(T)       
        return T

def arc_curvature(p0: np.ndarray, p1: np.ndarray, L: float) -> float:
    """
    给定二维圆弧的起点 p0、终点 p1 和弧长 L，返回曲率 k。
    
    参数
    ----
    p0 : (2,) array_like
        圆弧起点坐标 [x, y]
    p1 : (2,) array_like
        圆弧终点坐标 [x, y]
    L  : float
        圆弧的弧长（必须 > 0）
    
    返回
    ----
    k : float
        曲率（k = 1 / r）
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    
    # 弦长
    c = np.linalg.norm(p1 - p0)
    if c <= 0:
        raise ValueError("p0 和 p1 不能重合")
    if L <= c:
        return 0.0
    
    # 牛顿迭代求解 θ
    # 初值：用圆弧略大于弦长，θ 稍大于 0
    theta = 2.0 * np.sqrt(6 * (L / c - 1))  # 近似初值，稳定收敛
    
    def f(t):
        # 方程:  L * sin(t/2) = (c/2) * t
        return L * np.sin(t * 0.5) - 0.5 * c * t
    
    def df(t):
        # 导数
        return 0.5 * L * np.cos(t * 0.5) - 0.5 * c
    
    for _ in range(100):          # 牛顿迭代
        t_new = theta - f(theta) / df(theta)
        if abs(t_new - theta) < 1e-12:
            theta = t_new
            break
        theta = t_new
    else:
        raise RuntimeError("牛顿迭代未收敛")
    
    # 曲率
    k = theta / L
    return k

def signed_curvature(p0: np.ndarray,
                     t0: np.ndarray,
                     p1: np.ndarray) -> float:
    """
    已知圆弧起点 p0、起点单位切向 t0、终点 p1，
    返回带符号曲率 k：逆时针为正，顺时针为负。
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    t0 = np.asarray(t0, float)
    
    v = p1 - p0
    c = np.linalg.norm(v)
    if c == 0:
        return 0.0
    
    u = v / c
    d = np.clip(np.dot(t0, u), -1.0, 1.0)
    if d <= -1.0 + 1e-12:
        raise ValueError("切线反向，无法构成有限圆弧")
    
    phi = np.arccos(d)          # 0 ≤ φ < π
    cross = t0[0]*v[1] - t0[1]*v[0]
    sign = 1.0 if cross >= 0 else -1.0
    
    k = sign * (2.0 * np.sin(phi) / c)

    s = c * phi / np.sin(phi)

    return k, s


def _deep_copy_seg(seg: PCCSeg2D) -> PCCSeg2D:
    """深拷贝一段 PCC（含起点坐标）"""
    new = PCCSeg2D(seg_id=seg.seg_id, L=seg.L, kappa=seg.kappa, p_start=seg.p_start.copy())
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
    if seg.center is None or abs(k) < 1e-3:
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

import numpy as np

def point_at_frac(p_start, dir_start, k, frac, s):
    """
    给定圆弧起点 p_start、起点切向 dir_start、曲率 k 和段内比例 frac, 圆弧长度 s
    返回圆弧s的frac比例位置处的坐标
    """
    p_start = np.array(p_start, dtype=float)
    dir_start = np.array(dir_start, dtype=float)

    if k == 0:
        # 直线情况，直接按比例线性插值
        dir_start = dir_start / np.linalg.norm(dir_start)
        point = p_start + dir_start * s * frac
        return tuple(point)

    # 圆弧情况
    dir_start = dir_start / np.linalg.norm(dir_start)
    R = 1 / abs(k)

    # 垂直于切线的方向（逆时针方向）
    perp_dir = np.array([-dir_start[1], dir_start[0]])

    # 根据曲率符号确定圆心位置
    if k > 0:
        center = p_start + perp_dir * R
    else:
        center = p_start - perp_dir * R

    # 圆弧总角度
    theta_total = s * abs(k)
    # 当前角度
    theta_frac = theta_total * frac

    # 起点相对于圆心的角度
    start_vector = p_start - center
    start_angle = np.arctan2(start_vector[1], start_vector[0])

    # 当前角度
    if k > 0:
        current_angle = start_angle + theta_frac
    else:
        current_angle = start_angle - theta_frac

    # 当前位置坐标
    point = center + R * np.array([np.cos(current_angle), np.sin(current_angle)])

    return tuple(point)
