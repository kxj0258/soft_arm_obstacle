import mujoco
import numpy as np
from common.pcc import ContinuumArm2D, PCCSeg2D

# 在 sim_2d.py 里把 flex 相关 API 换成关节控制
# 在 sim_2d.py 里把关节控制换成 tendon 控制
class SoftArmSim2D:
    def __init__(self, model, data):
        self.model = model
        self.data  = data
        self.arm   = ArmModel2D.default()
        self.left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "left")
        self.right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "right")

    def step(self, action: np.ndarray):
        # action[0] 目标曲率
        kappa = self.arm.segs[0].kappa + action[0]*0.3
        self.arm.segs[0].kappa = np.clip(kappa, -3, 3)
        # 绳长差 → 张力差（简化：直接写 ctrl）
        L, d, k = 0.2, 0.06, self.arm.segs[0].kappa
        left_len  = L - 0.5*k*d
        right_len = L + 0.5*k*d
        # 归一化到 [-1,1]
        self.data.ctrl[0] = (left_len  - 0.2)*20
        self.data.ctrl[1] = (right_len - 0.2)*20

    def contacts(self):
        con = []
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            if name1 and name1.startswith("seg") and name2 and name2.startswith("obs"):
                con.append((g1, g2))
        return con

    def fk(self):
        return self.arm.fk()