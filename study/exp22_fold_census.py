#!/usr/bin/env python3
"""exp22: how many folds make one jump of the fibre size?

exp21 established that a fold splits *two* branches, with a square-root exponent.  The fibre size on
the SR0 -> SR5 path jumps from 8 to 16 (exp16), and by that exponent the jump cannot come from one
higher-order root producing eight branches: it should be **four separate folds**.

This experiment locates the events and counts the folds in each:

1. refine the parameter interval where the fibre size changes, by bisection on the *count*;
2. at the refined event, solve the current arm from scratch and look for configurations that are
   nearly singular (``sigma_min`` below a threshold) -- those are the coalescing pairs;
3. cluster them by proximity: one cluster per fold.  Four clusters at the 8 -> 16 jump confirms four
   double roots rather than one root of multiplicity four.

Run::

    python3 -m study.exp22_fold_census --pose-seed 0 --seeds 150
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at

#: Below this clearance a configuration is treated as a coalescing branch.
NEAR_SINGULAR = 0.02


def count_at(offset: float, s: float, target, seeds: int, rng) -> int:
    """Fibre size of the intermediate arm at ``s``."""
    model = sr0.robot(path_at(offset)(s))
    fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
    return fiber.count


def near_singular_clusters(offset: float, s: float, target, seeds: int, rng, tol: float = 0.15):
    """Configurations at ``s`` with small clearance, grouped into folds."""
    model = sr0.robot(path_at(offset)(s))
    fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
    circular = iks.circular_joints(model)
    points = [q for q in fiber.solutions if iks.sigma_min(model, q) < NEAR_SINGULAR]
    clusters: list[list[np.ndarray]] = []
    for point in points:
        for cluster in clusters:
            if iks.configuration_distance(point, cluster[0], circular) < tol:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return clusters, fiber.count


def main() -> int:
    """Locate the events and count the folds in each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--from", dest="s_from", type=float, default=0.60)
    parser.add_argument("--to", dest="s_to", type=float, default=0.72)
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr5_model = sr0.robot(link_transforms())
    rng_pose = np.random.default_rng(args.pose_seed)
    q_true = rng_pose.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    rng = np.random.default_rng(args.seed)

    grid = np.linspace(args.s_from, args.s_to, args.samples)
    counts = [count_at(offset, float(s), target, args.seeds, rng) for s in grid]
    print("fibre size across the interval:")
    print("  " + " ".join(f"{s:.3f}:{c}" for s, c in zip(grid, counts)))

    for index in range(len(grid) - 1):
        if counts[index] == counts[index + 1]:
            continue
        low, high = float(grid[index]), float(grid[index + 1])
        # bisection on the count: the size is a step function of s
        for _ in range(6):
            middle = 0.5 * (low + high)
            if count_at(offset, middle, target, args.seeds, rng) == counts[index]:
                low = middle
            else:
                high = middle
        clusters, count = near_singular_clusters(offset, high, target, args.seeds, rng)
        print(
            f"event between s = {grid[index]:.3f} ({counts[index]} solutions) and "
            f"s = {grid[index + 1]:.3f} ({counts[index + 1]}) -> located at s ~ {low:.4f} "
            f"| fibre just after: {count} | coalescing clusters (folds): {len(clusters)} "
            f"with clearances {[round(iks.sigma_min(sr0.robot(path_at(offset)(high)), c[0]), 4) for c in clusters]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
