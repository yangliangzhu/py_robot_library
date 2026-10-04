#!/usr/bin/env python3
"""exp47: fold seeds from a census of the whole discriminant, not from the tracked branches.

kimi measured the difference at its witness pose: seeds taken from tracked branches found two folds,
seeds taken from a census of configurations found four.  The reason is structural -- a tracked branch
only visits the part of the discriminant its own path crosses, while the *augmented system*
(``FK_s(q) = T`` together with ``det J_s(q) = 0``, seven equations in the seven unknowns ``(q, s)``)
can be seeded from any configuration and converges to *some* point of the discriminant, at whatever
parameter value it lives.

This compares the two sources pose by pose, so the study's "thirteen poses with an empty spectrum"
claim can be re-checked with the stronger instrument rather than left standing on the weaker one.

Run::

    python3 -m study.exp47_census_folds --poses 3 --samples 200
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import refine_fold
from study.exp41_fold_spectra import spectrum


def census_folds(offset: float, target, *, samples: int, seed: int, iterations: int = 8,
                 window: tuple[float, float] = (-0.1, 1.2)) -> list[tuple[float, np.ndarray]]:
    """Fold points found by seeding the augmented system from random configurations.

    Args:
        offset: The wrist-offset parameter of the real arm.
        target: The pose whose fibre is being studied.
        samples: Random configurations to seed from.
        seed: Random seed.
        iterations: Augmented-Newton iterations per seed.
        window: Parameter values to keep.

    Returns:
        Distinct ``(s, q)`` fold points.
    """
    path = path_at(offset)
    rng = np.random.default_rng(seed)
    model = sr0.robot(path(0.5))
    lower = np.asarray(model.lower_bounds)
    upper = np.asarray(model.upper_bounds)
    folds: list[tuple[float, np.ndarray]] = []
    for _ in range(samples):
        q0 = rng.uniform(lower, upper)
        s0 = rng.uniform(0.0, 1.0)
        s_star, q_star, residual = refine_fold(path, target, q0, s0, iterations=iterations)
        if residual > 1e-8 or not window[0] <= s_star <= window[1]:
            continue
        if not iks.within_limits(sr0.robot(path(s_star)), q_star):
            continue
        if any(
            abs(s_star - other) < 1e-4 and float(np.linalg.norm(q_star - q_other)) < 1e-3
            for other, q_other in folds
        ):
            continue
        folds.append((s_star, q_star))
    return sorted(folds)


def main() -> int:
    """Compare the two seed sources over several poses."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--target-seed", type=int, default=500)
    parser.add_argument("--near", type=float, default=0.05)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    lower = np.asarray(sr5_model.lower_bounds)
    upper = np.asarray(sr5_model.upper_bounds)
    census_extra = 0
    for pose in range(args.poses):
        rng = np.random.default_rng(args.target_seed + pose)
        target = sr5_model.fk(rng.uniform(lower, upper))
        seeds = [s.q for s in sr0.solve(sr0_model, target)]
        tracked = spectrum(offset, target, seeds, near=args.near, iterations=8)
        censused = census_folds(offset, target, samples=args.samples,
                                seed=args.target_seed + pose)
        extra = [
            fold for fold in censused
            if not any(abs(fold[0] - other[0]) < 1e-4
                       and float(np.linalg.norm(fold[1] - other[1])) < 1e-3 for other in tracked)
        ]
        census_extra += len(extra)
        print(f"pose {pose}: tracking source {len(tracked)} folds, census source {len(censused)} "
              f"folds, found only by the census {len(extra)}")
        if extra:
            print(f"    census-only folds at s = {[round(s, 6) for s, _q in extra]}")
    print(f"\ntotal census-only folds over {args.poses} poses: {census_extra} -- a positive number "
          f"means the tracking-source spectra were incomplete, as kimi measured at its witness pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
