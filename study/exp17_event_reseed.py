#!/usr/bin/env python3
"""exp17: event-driven reseeding -- completeness by scanning the path, not by following branches

exp16 established the mechanism: along the SR0 -> SR5 path the *fibre size* jumps (8 -> 16 -> 14 ->
12 -> 10 on one pose), deaths happen at the tracker's stalls and births happen elsewhere, to pairs
no tracked branch is following.  exp14 reseeded only at stalls -- that is, only at deaths -- which is
exactly why a residue of lost solutions survived.

This experiment takes the consequence seriously: **do not follow branches, scan the path.**
At every sample of ``s`` the intermediate arm is solved from scratch, all solutions found anywhere
along the sweep are used as seeds, and each is carried forward to the real arm.  If the union then
covers the real arm's fibre completely, the method is complete, and the price is the scan rather
than the tracking.

Run::

    python3 -m study.exp17_event_reseed --grid 21 --census-seeds 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp14_reseed_at_discriminant import path_from


def main() -> int:
    """Run the scan-and-continue experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seeds", type=int, nargs="+", default=[0, 1],
                        help="seeds for the target poses to test")
    parser.add_argument("--grid", type=int, default=21, help="samples along the parameter path")
    parser.add_argument("--census-seeds", type=int, default=300, help="local census seeds per sample")
    parser.add_argument("--reference-seeds", type=int, default=1000,
                        help="seeds for the reference census of the real arm")
    parser.add_argument("--match", type=float, default=1e-3, help="matching tolerance, radians")
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)

    for pose_seed in args.pose_seeds:
        pose_rng = np.random.default_rng(pose_seed)
        q_true = pose_rng.uniform(-2.0, 2.0, sr5_model.num_dof)
        target = sr5_model.fk(q_true)
        rng = np.random.default_rng(args.seed)

        # 1. scan: solve the intermediate arm at every sample, collect every solution as a seed
        started = time.perf_counter()
        seeds: list[tuple[float, np.ndarray]] = []
        counts: list[int] = []
        for s in np.linspace(0.0, 1.0, args.grid):
            model = sr0.robot(path_at(offset)(s))
            fiber = census(model, target, solver=make_lm_solver(model),
                           seeds=args.census_seeds, rng=rng)
            counts.append(fiber.count)
            for q in fiber.solutions:
                if not any(
                    iks.configuration_distance(q, other, circular) < args.match
                    for _s, other in seeds
                ):
                    seeds.append((float(s), q))
        scan_seconds = time.perf_counter() - started

        # 2. carry every seed forward to the real arm
        started = time.perf_counter()
        arrivals: list[np.ndarray] = []
        for s, q in seeds:
            path = path_at(offset) if s == 0.0 else path_from(offset, s)
            track = homotopy.track(path, target, q, sigma_floor=args.sigma_floor)
            if track.finished:
                arrivals.append(track.q)
        track_seconds = time.perf_counter() - started

        # 3. the reference: the real arm's fibre, solved from scratch
        fiber = census(sr5_model, target, solver=make_lm_solver(sr5_model),
                       seeds=args.reference_seeds, rng=rng)
        reference = fiber.solutions

        def matches(point: np.ndarray, others: list[np.ndarray]) -> bool:
            return any(
                iks.configuration_distance(point, other, circular) < args.match for other in others
            )

        unique_arrivals: list[np.ndarray] = []
        for q in arrivals:
            if not matches(q, unique_arrivals):
                unique_arrivals.append(q)
        missed = [q for q in reference if not matches(q, unique_arrivals)]
        extra = [q for q in unique_arrivals if not matches(q, reference)]
        print(
            f"pose seed {pose_seed}: fibre sizes along the path {counts} | "
            f"{len(seeds)} distinct seeds across {args.grid} samples | "
            f"{len(unique_arrivals)} distinct arrivals | reference census "
            f"{len(reference)} (from {fiber.converged}/{fiber.seeds}) | "
            f"reference solutions still missed: {len(missed)} | arrivals the census missed: "
            f"{len(extra)} | scan {scan_seconds:.1f} s, track {track_seconds:.1f} s"
        )
        for index, q in enumerate(missed):
            nearest = min(
                (iks.configuration_distance(q, other, circular) for other in unique_arrivals),
                default=float("nan"),
            )
            print(f"    missed {index}: clearance {iks.sigma_min(sr5_model, q):.3f}, "
                  f"nearest arrival {nearest:.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
