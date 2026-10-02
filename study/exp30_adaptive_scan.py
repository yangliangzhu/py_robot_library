#!/usr/bin/env python3
"""exp30: spending the scan's budget where the events are

The uniform scan (``complete_ik.solve_all(mode="scan")``) solves the intermediate arm at every grid
point, and exp29 showed that most of that budget is wasted: 5 x 80 already reached everything 15 x 200
did.  The fibre size only changes at *events*, so the budget belongs in the intervals where it changes.

This experiment does a two-pass scan and compares it with the uniform one:

1. a coarse grid with a small census per point;
2. for every adjacent pair of grid points whose **solution counts differ**, insert two midpoints with a
   larger census, because that interval contains at least one event;
3. carry everything forward and deduplicate, exactly as the uniform version does.

The claim to check is not "faster" but "**same solutions, less time**" -- and, importantly, whether the
count-difference trigger finds the intervals that matter (a birth inside one interval is visible as a
count change at its ends, which is what makes the trigger sound).

Run::

    python3 -m study.exp30_adaptive_scan --pose-seeds 0 7 --uniform-grid 9 --uniform-seeds 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from model import IkType, ModelFactory
from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.complete_ik import solve_all
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def carry_forward(offset: float, target, s: float, seeds, *, sigma_floor: float) -> list[np.ndarray]:
    """Solve the arm at ``s`` and carry every solution to the real arm."""
    path = path_at(offset)
    intermediate = sr0.robot(path(s))
    local = census(intermediate, target, solver=make_lm_solver(intermediate), seeds=seeds,
                   rng=np.random.default_rng(int(s * 1000)))
    arrivals = []
    for q in local.solutions:
        track = homotopy.track(path if s == 0.0 else _from(path, s), target, q,
                               sigma_floor=sigma_floor)
        if track.finished:
            arrivals.append(track.q)
    return arrivals


def _from(path, s0: float):
    """The path restricted to ``[s0, 1]``, reparameterised to ``[0, 1]``."""
    def make(t: float) -> list[np.ndarray]:
        return path(s0 + t * (1.0 - s0))

    return make


def adaptive_scan(offset: float, target, circular, *, coarse: int, coarse_seeds: int,
                  fine_seeds: int, sigma_floor: float) -> tuple[list[np.ndarray], list[float]]:
    """Two-pass scan: coarse grid, then refine the intervals whose counts differ."""
    samples = list(np.linspace(0.0, 1.0, coarse))
    counts, found = [], []
    for s in samples:
        arrivals = carry_forward(offset, target, float(s), coarse_seeds, sigma_floor=sigma_floor)
        counts.append(len(arrivals))
        found.extend(arrivals)
    refined: list[float] = []
    for index in range(len(samples) - 1):
        if counts[index] != counts[index + 1]:
            left, right = samples[index], samples[index + 1]
            refined.extend([left + (right - left) / 3.0, left + 2.0 * (right - left) / 3.0])
    for s in refined:
        found.extend(carry_forward(offset, target, float(s), fine_seeds, sigma_floor=sigma_floor))
    unique: list[np.ndarray] = []
    for q in found:
        if not any(iks.configuration_distance(q, other, circular) < 1e-3 for other in unique):
            unique.append(q)
    return unique, counts


def main() -> int:
    """Compare the adaptive scan with the uniform one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seeds", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--coarse-seeds", type=int, default=60)
    parser.add_argument("--fine-seeds", type=int, default=120)
    parser.add_argument("--uniform-grid", type=int, default=9)
    parser.add_argument("--uniform-seeds", type=int, default=300)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    for pose_seed in args.pose_seeds:
        rng = np.random.default_rng(pose_seed)
        target, _ = iks.random_reachable_target(model, rng)

        started = time.perf_counter()
        adaptive, counts = adaptive_scan(offset, target, circular, coarse=args.coarse,
                                         coarse_seeds=args.coarse_seeds,
                                         fine_seeds=args.fine_seeds, sigma_floor=2e-3)
        adaptive_seconds = time.perf_counter() - started

        started = time.perf_counter()
        uniform = solve_all(model, target, mode="scan", grid=args.uniform_grid,
                            seeds=args.uniform_seeds, rng=np.random.default_rng(pose_seed))
        uniform_seconds = time.perf_counter() - started
        uniform_points = [solution.q for solution in uniform]

        def misses(reference, other) -> int:
            return sum(
                1 for q in reference
                if not any(iks.configuration_distance(q, o, circular) < 1e-3 for o in other)
            )

        print(
            f"pose seed {pose_seed}: coarse counts {counts} | adaptive {len(adaptive)} solutions "
            f"in {adaptive_seconds:5.1f} s | uniform {len(uniform_points)} solutions in "
            f"{uniform_seconds:5.1f} s | adaptive misses {misses(uniform_points, adaptive)} of "
            f"uniform, uniform misses {misses(adaptive, uniform_points)} of adaptive"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
