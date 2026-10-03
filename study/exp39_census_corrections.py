#!/usr/bin/env python3
"""exp39: how much does a census over-count?  Representatives versus torus classes.

Every revolute joint is 2 pi periodic in the kinematics, so two representatives differing by multiples
of 2 pi are *one* solution; and a representative outside the joint limits can still be a solution whose
torus class has an admissible member.  Both distinctions were found late in this study (a "9 solution"
pose that is 8, a control arm reporting 15 for a spherical wrist bounded by 8), so this experiment
measures how often they matter, over several poses.

Run::

    python3 -m study.exp39_census_corrections --poses 6 --seeds 600
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver


def main() -> int:
    """Print the correction table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--floor", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    rng = np.random.default_rng(args.seed)
    print(f"{args.robot}: representatives | torus classes | physically admissible classes | "
          f"representatives outside the limits")
    totals = [0, 0, 0, 0]
    for pose in range(args.poses):
        target, _ = iks.random_reachable_target(model, rng)
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        kept = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.floor]
        classes: list[np.ndarray] = []
        for candidate in kept:
            if not any(iks.torus_distance(candidate, other) < 1e-3 for other in classes):
                classes.append(candidate)
        admissible = [c for c in classes if iks.physically_admissible(model, c)]
        outside = sum(1 for q in kept if not iks.within_limits(model, q))
        totals[0] += len(kept)
        totals[1] += len(classes)
        totals[2] += len(admissible)
        totals[3] += outside
        flag = ""
        if len(kept) != len(classes):
            flag += "  <-- torus over-count"
        if len(classes) != len(admissible):
            flag += "  <-- limits"
        print(f"  pose {pose}: {len(kept):3d} | {len(classes):3d} | {len(admissible):3d} | "
              f"{outside:3d}{flag}")
    print(f"totals: {totals[0]} representatives -> {totals[1]} torus classes "
          f"({totals[0] - totals[1]} duplicates) -> {totals[2]} physically admissible "
          f"({totals[1] - totals[2]} classes without an in-limit representative), "
          f"{totals[3]} representatives outside the limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
