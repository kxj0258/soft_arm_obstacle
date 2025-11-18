import mujoco
import numpy as np
from .model_3d import ArmModel3D

class SoftArmSim3D:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.flexid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "arm")
        self.arm = ArmModel3D.default()
        self._update_constraints()

    def _update_constraints(self):
        """把 κ1,κ2 映射成 4 根绳长 → margin"""
        seg = self.arm.segs[0]
        L, k1, k2 = seg.L, seg.kappa1, seg.kappa2
        d = 0.06          # 绳到中心距离
        # 单位长度伸长量 ≈ ± 0.5*d*(k1*sign_y + k2*sign_z)
        lenU = L - 0.5*d*( k1 + k2)
        lenD = L - 0.5*d*(-k1 - k2)
        lenL = L - 0.5*d*( k1 - k2)
        lenR = L - 0.5*d*(-k1 + k2)
        rest = 1.0
        for i, l in enumerate([lenU, lenD, lenL, lenR]):
            self.data.eq_margin[i] = l - rest

    def step(self, action: np.ndarray):
        """action: [dk1, dk2]"""
        k1 = self.arm.segs[0].kappa1 + action[0]*0.3
        k2 = self.arm.segs[0].kappa2 + action[1]*0.3
        self.arm.segs[0].kappa1 = np.clip(k1, -3, 3)
        self.arm.segs[0].kappa2 = np.clip(k2, -3, 3)
        self._update_constraints()
        mujoco.mj_step(self.model, self.data)

    def contacts(self):
        con = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.flex[0] != -1:
                con.append((c.flex_node[0], c.geom2))
        return con

    def insert_node_at(self, node_id: int):
        mujoco.mj_flexInsert(self.model, self.data, self.flexid, node_id)
        # 简单拆段
        seg = self.arm.segs[0]
        frac = 0.5
        L1, L2 = seg.L*frac, seg.L*(1-frac)
        self.arm.segs[0:1] = [PCCSeg3D(L1, seg.kappa1, seg.kappa2),
                              PCCSeg3D(L2, seg.kappa1*0.8, seg.kappa2*0.8)]

    def fk(self):
        T = self.arm.fk()
        pos = T[:3,3]
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(T[:3,:3]).as_quat()  # [x,y,z,w]
        return np.concatenate([pos, quat])