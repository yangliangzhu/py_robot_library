#!/usr/bin/env python3
"""exp45: is SR0 cuspidal?  If so, it witnesses the orthogonality of solvability and branch structure.

The study's third answer splits into two orthogonal questions:

* **solvability in radicals** -- a property of the complex monodromy group (the SR5's is non-solvable,
  so its IK has no radical solution);
* **branch structure** -- a property of the real configuration space (cuspidality: two solutions of one
  pose joined by a path that avoids the singular set).

A single arm that is *both* closed-form solvable *and* cuspidal would witness the orthogonality
directly.  SR0 is the natural candidate: it is the SR5 with link 5's wrist offset removed, every
solution comes from a chain of ``+-acos`` choices (study/sr0.py solves it deterministically in closed
form), so it is solvable by radicals by construction.  The question is whether it is also cuspidal --
whether any two of its solutions are connected while keeping clearance.

Run::

    python3 -m study.exp45_sr0_cuspidality --seeds 600
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.chamber import resolve_pair


def main() -> int:
    """Measure SR0's fibre, its sign groups and the clearance-keeping connections within them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--floor", type=float, default=2e-3)
    parser.add_argument("--target-seed", type=int, default=700)
    parser.add_argument("--poses", type=int, default=1)
    args = parser.parse_args()

    model = sr0.robot()
    circular = iks.circular_joints(model)
    cuspidal_poses = 0
    total_connected = 0
    for pose in range(args.poses):
        result = _one_pose(model, circular, args, args.target_seed + pose)
        cuspidal_poses += int(result["connected"] > 0)
        total_connected += result["connected"]
    if args.poses > 1:
        print(f"\nover {args.poses} poses: cuspidal in {cuspidal_poses}, "
              f"total within-sign connections {total_connected}")
    return 0 if cuspidal_poses or args.poses == 1 else 0


def _one_pose(model, circular, args, target_seed: int) -> dict:
    """Measure one pose; returns the numbers the tally needs."""
    rng = np.random.default_rng(target_seed)
    lower = np.asarray(model.lower_bounds)
    upper = np.asarray(model.upper_bounds)
    target = model.fk(rng.uniform(lower, upper))
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions: list[np.ndarray] = []
    for candidate in fiber.solutions:
        if iks.sigma_min(model, candidate) <= args.floor:
            continue
        if not iks.physically_admissible(model, candidate):
            continue
        if any(iks.torus_distance(candidate, other) < 1e-3 for other in solutions):
            continue
        solutions.append(candidate)
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]

    connected = 0
    tested = 0
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            if signs[i] != signs[j]:
                continue  # the sign certificate already proves these differ
            tested += 1
            if resolve_pair(model, solutions[i], solutions[j], delta=args.delta,
                            circular=circular).connected:
                connected += 1
    print(f"  pose seed {target_seed}: {len(solutions)} solutions "
          f"({signs.count(1)}+/{signs.count(-1)}-), closed form {len(sr0.solve(model, target))} | "
          f"within-sign pairs {tested}, connected {connected}")
    return {"solutions": len(solutions), "connected": connected}


if __name__ == "__main__":
    raise SystemExit(main())
