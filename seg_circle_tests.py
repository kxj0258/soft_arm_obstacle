import numpy as np
from common.pcc import PCCSeg2D
from tasks.arm2d.model_2d import seg_circle_nearest

np.set_printoptions(precision=6, suppress=True)

# helper to build a segment with given curvature and sweep angle
def make_seg(k, total_ang, p_start=np.array([0.0, 0.0])):
    if abs(k) < 1e-9:
        L = 1.0
    else:
        L = total_ang / k
    seg = PCCSeg2D(L=L, kappa=k, p_start=p_start.copy())
    return seg


def run_case(name, seg, c, r):
    print(f"\n--- {name} ---")
    print("seg.p_start=", seg.p_start)
    print("seg.p_end=", seg.p_end)
    print("seg.center=", seg.center, "R=", abs(1.0/seg.kappa) if abs(seg.kappa)>1e-12 else None)
    print("obstacle center=", c, "r=", r)
    res = seg_circle_nearest(seg, c, r)
    print("result=", res)


if __name__ == '__main__':
    # Case A: no collision (far away)
    seg = make_seg(0.2, np.pi/2)
    c_far = seg.center + np.array([20.0, 0.0])
    r = 1.0
    run_case("No collision (far)", seg, c_far, r)

    # Case B: tangent external
    seg = make_seg(0.2, np.pi/2)
    R = abs(1.0/seg.kappa)
    vec_mid = (seg.p_end + seg.p_start) / 2.0 - seg.center
    ndir = vec_mid / (np.linalg.norm(vec_mid) + 1e-12)
    r = 1.0
    c_tangent = seg.center + ndir * (R + r)
    run_case("Tangent external", seg, c_tangent, r)

    # Case C: intersecting (overlap)
    seg = make_seg(0.2, np.pi/2)
    R = abs(1.0/seg.kappa)
    ndir = ((seg.p_end + seg.p_start) / 2.0 - seg.center)
    ndir = ndir / (np.linalg.norm(ndir) + 1e-12)
    r = 1.5
    c_inter = seg.center + ndir * (R + r - 0.5)
    run_case("Intersecting", seg, c_inter, r)

    # Case D: endpoint inside
    seg = make_seg(0.2, np.pi/2)
    r = 2.0
    c_end = seg.p_end.copy()
    run_case("Endpoint inside", seg, c_end, r)

    # Case E: centers coincide, r >= R
    seg = make_seg(0.2, np.pi/2)
    r = abs(1.0/seg.kappa) + 0.5
    c_coincide = seg.center.copy()
    run_case("Centers coincide covers circle", seg, c_coincide, r)
