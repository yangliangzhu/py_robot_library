#!/usr/bin/env python3
"""exp40: is the Raghavan-Roth bound attained?  Sixteen real solutions on an intermediate arm.

The fibre-size sweep of exp16 reported sixteen solutions at s ~ 0.70 along the SR0 path, which would
attain the classical bound for a general 6R (a spherical wrist is bounded by eight).  That number was
measured before this study learned to (a) identify solutions on the torus and (b) separate physical
admissibility from the pose equation, so it has to be re-measured with both.

Run::

    python3 -m study.exp40_sixteen --seeds 800
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def main() -> int:
    """Count the fibre of intermediate arms with all three filters."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=float, nargs="+", default=[0.60, 0.66, 0.70, 0.75, 0.90, 1.0])
    parser.add_argument("--seeds", type=int, default=800)
    parser.add_argument("--floor", type=float, default=2e-3)
    parser.add_argument("--pose-seed", type=int, default=0)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr5 = sr0.robot(link_transforms())
    rng_pose = np.random.default_rng(args.pose_seed)
    q_true = rng_pose.uniform(-2.0, 2.0, sr5.num_dof)
    target = sr5.fk(q_true)
    print("s | representatives | torus classes | physically admissible | min clearance of the classes")
    best = 0
    for s in args.samples:
        model = sr0.robot(path_at(offset)(s))
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds,
                       rng=np.random.default_rng(int(1000 * s)))
        kept = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.floor]
        classes: list[np.ndarray] = []
        for candidate in kept:
            if not any(iks.torus_distance(candidate, other) < 1e-3 for other in classes):
                classes.append(candidate)
        admissible = [c for c in classes if iks.physically_admissible(model, c)]
        clearance = min((iks.sigma_min(model, c) for c in classes), default=float("nan"))
        best = max(best, len(classes))
        print(f"{s:4.2f} | {len(kept):3d} | {len(classes):3d} | {len(admissible):3d} | {clearance:.3f}")
    print(f"\nlargest torus-distinct fibre along the path: {best} "
          f"(Raghavan-Roth bound 16, spherical-wrist bound 8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
