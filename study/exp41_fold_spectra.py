#!/usr/bin/env python3
"""exp41: fold spectra over several poses -- how many events does a path to the solvable arm cross?

exp23 identified the seven folds of one pose and exp25 verified the +-2 law on its windows.  This
experiment repeats the enumeration over several poses to see how typical that picture is: how many
folds a path crosses, and whether the count varies as much as the fibre sizes do.

Run::

    python3 -m study.exp41_fold_spectra --poses 3
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import approach, refine_fold


def spectrum(offset: float, target, seeds, *, near: float, iterations: int):
    """Deterministic fold spectrum: refine every near-singular configuration on the tracked branches."""
    path = path_at(offset)
    candidates = []
    for seed in seeds:
        for s, q in approach(path, target, seed, 1.0):
            if iks.sigma_min(sr0.robot(path(s)), q) < near:
                candidates.append((s, q))
    folds: list[tuple[float, np.ndarray]] = []
    for s, q in candidates:
        s_star, q_star, residual = refine_fold(path, target, q, s, iterations=iterations)
        if residual > 1e-8:
            continue
        if not any(
            abs(s_star - other) < 1e-4 and float(np.linalg.norm(q_star - q_other)) < 1e-3
            for other, q_other in folds
        ):
            folds.append((s_star, q_star))
    return sorted(folds)


def main() -> int:
    """Enumerate the spectra and print a compact table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--target-seed", type=int, default=200)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    total = 0
    fold_free = 0
    fold_counts: list[int] = []
    for pose in range(args.poses):
        target, _ = iks.random_reachable_target(
            sr5_model, np.random.default_rng(args.target_seed + pose)
        )
        seeds = [s.q for s in sr0.solve(sr0_model, target)]
        folds = spectrum(offset, target, seeds, near=args.near, iterations=args.iterations)
        total += len(folds)
        fold_counts.append(len(folds))
        fold_free += int(not folds)
        values = [round(s, 6) for s, _q in folds]
        print(f"pose {pose}: {len(folds)} folds at {values}")
    print(f"\ntotal: {total} folds over {args.poses} poses | counts {fold_counts} | "
          f"fold-free {fold_free}/{args.poses} ({100.0 * fold_free / max(args.poses, 1):.0f}%)")
    print("interpretation: a fold-free pose is one where every branch survives the straight path, which "
          "is exactly where the 0.5 s homotopy mode is already complete -- the expensive scan is "
          "insurance for the poses that have folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
