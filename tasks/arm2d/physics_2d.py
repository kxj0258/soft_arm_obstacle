import numpy as np
from typing import Optional, Tuple

import numpy as np
from typing import Optional, Tuple

from common.pcc import PCCSeg2D



def resolve_impulse(arm, obstacles, dt, k=1e4, d=1e2):
    """对每段圆弧做弹簧-阻尼冲量，并拆段"""
    segs = arm.segs_obs  # 只检测 obs 链
    for cx, cy, r in obstacles:
        for i, seg in enumerate(segs):
            info = seg_circle_nearest(seg, np.array([cx, cy]), r)
            if info:
                p, n, depth, col_point, frac = info   # 多收一个碰撞点
                # 把圆上碰撞点一并传给 arm
                arm.insert_segment_at(i, frac, n, col_point)
                break        # 同一障碍只处理一次