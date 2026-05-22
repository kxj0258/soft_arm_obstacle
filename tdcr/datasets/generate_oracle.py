from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tdcr.robot_params import TDCRRobotParams
from tdcr.shapes import Obstacle3D
from tdcr.solvers import CosseratStaticSolver, PCCAContact3DSolver, TDCRModelingOracleSolver

from .schema import DEFAULT_MAX_CONTACTS, DEFAULT_MAX_OBSTACLES, collate_samples, sample_from_shape


def _sample_obstacles(
    rng: np.random.Generator,
    *,
    count: int,
    total_length: float,
) -> list[Obstacle3D]:
    obstacles: list[Obstacle3D] = []
    for _ in range(int(count)):
        radius = float(rng.uniform(0.010, 0.025))
        center = np.array(
            [
                rng.uniform(-0.065, 0.065),
                rng.uniform(-0.065, 0.065),
                rng.uniform(0.04, total_length - 0.025),
            ],
            dtype=float,
        )
        obstacles.append(Obstacle3D(center=center, radii=radius, kind="sphere"))
    return obstacles


def generate_dataset(
    *,
    output_path: str | Path,
    num_samples: int,
    seed: int = 0,
    tau_max: float = 3.5,
    obstacle_count: int = 3,
    backend: str = "pcca_contact_3d",
) -> Path:
    params = TDCRRobotParams()
    if backend == "oracle":
        solver = TDCRModelingOracleSolver(params)
    elif backend == "cosserat_static":
        solver = CosseratStaticSolver(params, modelling_approach="VC_Ref")
    elif backend == "pcca_contact_3d":
        solver = PCCAContact3DSolver(params)
    else:
        raise ValueError("backend must be 'pcca_contact_3d', 'cosserat_static', or 'oracle'.")

    rng = np.random.default_rng(seed)
    samples = []
    for index in range(int(num_samples)):
        tau = rng.uniform(0.0, float(tau_max), size=(params.num_segments, params.tendons_per_segment))
        obstacles = _sample_obstacles(rng, count=obstacle_count, total_length=params.total_length)
        shape = solver.solve(tau=tau, obstacles=obstacles)
        samples.append(
            sample_from_shape(
                shape,
                robot_params=params,
                obstacles=obstacles,
                max_contacts=DEFAULT_MAX_CONTACTS,
                max_obstacles=DEFAULT_MAX_OBSTACLES,
                metadata={"sample_index": index, "backend": backend},
            )
        )

    batch = collate_samples(samples)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **batch.as_dict())
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate contact-aware 3D TDCR oracle data.")
    parser.add_argument("--output-path", type=str, default="data/tdcr3d_oracle_samples.npz")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-max", type=float, default=3.5)
    parser.add_argument("--obstacle-count", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=("pcca_contact_3d", "cosserat_static", "oracle"),
        default="pcca_contact_3d",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = generate_dataset(
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        tau_max=args.tau_max,
        obstacle_count=args.obstacle_count,
        backend=args.backend,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
