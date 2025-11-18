from common.pcc import PCCSeg3D, ContinuumArm3D
from typing import List

class ArmModel3D(ContinuumArm3D):
    """仅封装一层，方便以后加 calibration"""
    def __init__(self, segs: List[PCCSeg3D]):
        super().__init__(segs)
    @staticmethod
    def default():
        return ArmModel3D([PCCSeg3D(L=1.0, kappa1=0.0, kappa2=0.0)])