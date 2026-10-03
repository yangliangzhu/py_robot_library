#!/usr/bin/env python3
"""exp42: how often does the cheap homotopy mode suffice?  A survival census over poses.

exp41 found ten sampled poses with no folds at all, which the two pose distributions do not explain.
The natural reading is that for those poses every SR0 branch survives the straight parameter path, so
the fibre size never changes and there is nothing to die at.  That is the practically important
statistic: the 0.5 s homotopy mode is complete exactly for such poses, and the expensive scan is
insurance for the rest.

This measures the distribution: for each pose, carry SR0's eight closed-form solutions along
``link 5 y: 0 -> 0.136 m`` and count how many arrive.

Run::

    python3 -m study.exp42_survival --poses 12
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def main() -> int:
    """Tally how many branches survive per pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=12)
    parser.add_argument("--target-seed", type=int, default=500)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    lower = np.asarray(sr5_model.lower_bounds)
    upper = np.asarray(sr5_model.upper_bounds)
    arrivals: list[int] = []
    for pose in range(args.poses):
        rng = np.random.default_rng(args.target_seed + pose)
        target = sr5_model.fk(rng.uniform(lower, upper))
        seeds = sr0.solve(sr0_model, target)
        arrived = 0
        for seed in seeds:
            track = homotopy.track(path, target, seed.q, sigma_floor=args.sigma_floor)
            arrived += int(track.finished)
        arrivals.append(arrived)
        print(f"pose {pose}: {arrived}/{len(seeds)} branches arrive")
    print(f"\narrivals {arrivals}")
    print(f"poses where every SR0 branch survives (fold-free, cheap mode complete): "
          f"{sum(1 for v, n in zip(arrivals, [8] * args.poses) if v == n)}/{args.poses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
