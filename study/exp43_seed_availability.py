#!/usr/bin/env python3
"""exp43: how often can the SR0-seeded homotopy even start?  A seed-availability census.

exp42's twelve-pose survival census contained one pose with zero arrivals, and exp41 showed why: the
SR0 solver returns *no* solutions for that pose, so there are no seeds to carry.  That is not a branch
dying -- it is the method's applicability boundary, since SR0 is a different arm (13.6 cm of wrist
offset removed) with a different reachable workspace, and a pose in the SR5's workspace need not be in
SR0's.

This measures the frequency, and with it how often Finding 2's recipe is applicable at all.

Run::

    python3 -m study.exp43_seed_availability --poses 20
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.exp11_dh_continuation import link_transforms


def main() -> int:
    """Count poses with no SR0 seeds, and their distribution over pose sources."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=20)
    parser.add_argument("--target-seed", type=int, default=500)
    parser.add_argument("--mode", choices=("fk", "reachable"), default="fk")
    args = parser.parse_args()

    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    lower = np.asarray(sr5_model.lower_bounds)
    upper = np.asarray(sr5_model.upper_bounds)
    counts: list[int] = []
    for pose in range(args.poses):
        rng = np.random.default_rng(args.target_seed + pose)
        if args.mode == "fk":
            target = sr5_model.fk(rng.uniform(lower, upper))
        else:
            target, _ = iks.random_reachable_target(sr5_model, rng)
        counts.append(len(sr0.solve(sr0_model, target)))
    empty = sum(1 for value in counts if value == 0)
    print(f"mode {args.mode}, {args.poses} poses: SR0 seed counts {counts}")
    print(f"poses where the SR0-seeded method cannot start (zero seeds): {empty}/{args.poses} "
          f"({100.0 * empty / args.poses:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
