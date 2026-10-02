#!/usr/bin/env python3
"""exp29: how cheap can the complete mode be?  A budget sweep of the scan

``complete_ik.solve_all(mode="scan")`` is the only mode that was complete in every experiment of this
study, and it is the expensive one (about 40-60 s): it solves the intermediate arm on a grid of
parameters and carries every solution forward.  Its two knobs are the grid size and the seeds per grid
point, so this experiment sweeps them on the same pose and reports what each budget finds, how long it
takes, and which solutions a cheaper budget misses relative to the richest one.

That turns "the scan is complete but slow" into a usable trade-off curve: the cheapest setting that
still finds everything on a hard pose.

Run::

    python3 -m study.exp29_scan_budget --pose-seeds 0
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.complete_ik import solve_all

#: (grid, seeds per grid point) budgets, cheapest first.
BUDGETS = ((3, 60), (5, 80), (5, 200), (9, 120), (9, 300), (15, 200))


def main() -> int:
    """Sweep the scan budget on one pose per seed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seeds", type=int, nargs="+", default=[0, 7])
    args = parser.parse_args()

    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    for pose_seed in args.pose_seeds:
        rng = np.random.default_rng(pose_seed)
        target, _ = iks.random_reachable_target(model, rng)
        results = {}
        print(f"pose seed {pose_seed}: scan budget sweep")
        for grid, seeds in BUDGETS:
            started = time.perf_counter()
            solutions = solve_all(model, target, mode="scan", grid=grid, seeds=seeds,
                                  rng=np.random.default_rng(pose_seed))
            seconds = time.perf_counter() - started
            results[(grid, seeds)] = [solution.q for solution in solutions]
            print(f"  grid {grid:2d} x {seeds:3d} seeds: {len(solutions):2d} solutions "
                  f"in {seconds:5.1f} s")
        richest = max(results, key=lambda key: len(results[key]))
        reference = results[richest]
        print(f"  reference = the richest budget {richest} with {len(reference)} solutions")
        for key, found in results.items():
            missed = sum(
                1 for q in reference
                if not any(iks.configuration_distance(q, o, circular) < 1e-3 for o in found)
            )
            print(f"    grid {key[0]:2d} x {key[1]:3d} seeds misses {missed:2d} of the reference "
                  f"({len(found)} of {len(reference)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
