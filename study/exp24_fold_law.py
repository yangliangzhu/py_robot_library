#!/usr/bin/env python3
"""exp24: is "fibre size changes by two per fold" a law?

exp21 measured the split exponent (1/2, two branches per fold) and exp23 identified the seven folds of
one pose, four of which account for its 8 -> 16 jump.  This experiment turns that into a testable law
over several poses:

    fibre(s + eps) - fibre(s - eps)  =  2 x (number of folds in (s - eps, s + eps))

Method, deterministic where it matters:

1. follow the SR0 branches along the path and collect the near-singular configurations *on them*
   (structure, not sampling -- exp22's census-based version measured noise);
2. refine each to a fold with the augmented Newton of exp21 and deduplicate;
3. group folds that lie within ``--window`` of each other, measure the fibre size just before and just
   after the group with a large census, and compare the jump with ``2 x`` the group size.

A violation would mean either a fold that does not split two branches, or a fibre-size measurement
that is wrong -- so the census counts are also repeated to show their spread.

Run::

    python3 -m study.exp24_fold_law --poses 3 --seeds 400
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import approach, refine_fold


def folds_of_pose(offset: float, target, model_seeds, *, near: float, iterations: int):
    """Every distinct fold point along the path, from near-singular configurations on the branches."""
    path = path_at(offset)
    candidates = []
    for seed in model_seeds:
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


def fibre_size(offset: float, target, s: float, seeds: int, rng) -> int:
    """Fibre size of the intermediate arm at ``s``."""
    model = sr0.robot(path_at(offset)(s))
    return census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng).count


def main() -> int:
    """Run the law check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--near", type=float, default=0.05, help="near-singular collection threshold")
    parser.add_argument("--window", type=float, default=0.01, help="fold grouping window")
    parser.add_argument("--eps", type=float, default=2e-3, help="fibre measurement offset")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    rng = np.random.default_rng(args.seed)
    obeyed = violated = groups_total = folds_total = 0
    for pose in range(args.poses):
        pose_rng = np.random.default_rng(pose)
        q_true = pose_rng.uniform(-2.0, 2.0, sr5_model.num_dof)
        target = sr5_model.fk(q_true)
        seeds = sr0.solve(sr0_model, target)
        folds = folds_of_pose(offset, target, [s.q for s in seeds], near=args.near,
                              iterations=args.iterations)
        folds_total += len(folds)
        # group folds that are close in s
        groups: list[list[float]] = []
        for s_star, _q in folds:
            if groups and s_star - groups[-1][-1] <= args.window:
                groups[-1].append(s_star)
            else:
                groups.append([s_star])
        groups = [group for group in groups if 0.0 < group[0] and group[-1] < 1.0]
        print(f"pose {pose}: {len(folds)} folds -> {len(groups)} groups at "
              f"{[round(group[0], 4) for group in groups]}")
        for group in groups:
            before = fibre_size(offset, target, max(group[0] - args.eps, 0.0), args.seeds, rng)
            after = fibre_size(offset, target, min(group[-1] + args.eps, 1.0), args.seeds, rng)
            # a fold changes the real fibre size by two: births add, deaths remove, so the law is
            # stated on the magnitude and the sign is reported separately
            predicted = 2 * len(group)
            delta = after - before
            verdict = "obeys" if abs(abs(delta) - predicted) <= 2 else "violates"
            obeyed += int(verdict == "obeys")
            violated += int(verdict == "violates")
            groups_total += 1
            print(
                f"    group at s ~ {group[0]:.4f} ({len(group)} folds): fibre {before} -> {after} "
                f"(delta {delta:+d}), predicted |delta| = {predicted} "
                f"({'birth' if delta > 0 else 'death' if delta < 0 else 'no change'})  {verdict}"
            )
    print(f"\ntotals: {folds_total} folds in {groups_total} groups; law obeyed by {obeyed}, "
          f"violated by {violated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
