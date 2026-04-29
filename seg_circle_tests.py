import numpy as np
from common.pcc import PCCSegment2D
from tasks.arm2d.physics_2d import CircleObstacle, detect_collision_event, summarize_contacts

np.set_printoptions(precision=6, suppress=True)

def make_segment(kappa: float, length: float = 10.0) -> PCCSegment2D:
    return PCCSegment2D(length=length, kappa=kappa)


def run_case(name, segment, center, radius, arm_radius=1.0):
    print(f"\n--- {name} ---")
    print("segment.start=", segment.start_position)
    print("segment.end=", segment.end_position)
    obstacle = CircleObstacle(center=np.asarray(center, dtype=float), radius=radius)
    event = detect_collision_event(
        segment=segment,
        obstacle=obstacle,
        obstacle_index=0,
        segment_index=0,
        arm_radius=arm_radius,
    )
    print("obstacle center=", obstacle.center, "r=", obstacle.radius, "arm_radius=", arm_radius)
    print("event=", event)
    summary = summarize_contacts(
        segments=[segment],
        obstacles=[obstacle],
        arm_radius=arm_radius,
        collision_events=[] if event is None else [event],
    )
    print("contact_summary=", summary)


if __name__ == "__main__":
    segment = make_segment(0.0, length=12.0)
    run_case("No collision", segment, center=np.array([30.0, 0.0]), radius=2.0)

    segment = make_segment(0.0, length=12.0)
    run_case("Straight segment collision", segment, center=np.array([8.0, 1.0]), radius=2.2)

    segment = make_segment(0.12, length=14.0)
    run_case("Curved segment collision", segment, center=np.array([11.0, 2.5]), radius=2.0)
