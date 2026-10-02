#!/usr/bin/env python3
"""exp26: the refined law over several poses -- the fibre changes by two per *real event*

exp25 refined the "two solutions per fold" law: only a fold whose coalescing pair crosses between real
and complex changes the real fibre size; a fold of an already-complex pair (a turning point) deforms
the fibre without changing what is real.  That makes a testable per-fold prediction:

* classify each fold by probing the fibre just before and just after it -- an **event** is a fold with
  a real pair on exactly one side (birth if after, death if before), an **invisible** fold has no real
  pair near it on either side;
* predict ``delta = +-2 x (number of events)`` for the window, with the sign from the events;
* measure ``delta`` deterministically (exp25: survivors carried + births found), which has no sampling
  noise in the survivor part.

Then report the tally and every disagreement, because a law that is only checked where it holds is not
checked at all.

Run::

    python3 -m study.exp26_fold_law_multi --poses 3 --seeds 300
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
from study.exp25_deterministic_count import births_at, carry


def folds_of(offset: float, target, seeds, *, near: float, iterations: int):
    """Distinct fold points along the path, from near-singular configurations on tracked branches."""
    path = path_at(offset)
    candidates = []
    for seed in seeds:
        for s, q in approach(path, target, seed, 1.0):
            if iks.sigma_min(sr0.robot(path(s)), q) < near:
                candidates.append((s, q))
    folds = []
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


def classify_fold(offset: float, target, s_star: float, q_star: np.ndarray, circular, *,
                  eps: float, seeds: int, rng, radius: float = 0.1) -> str:
    """``'birth'``, ``'death'`` or ``'invisible'`` -- whether the pair is real on one side only."""
    sides = {}
    for label, s in (("before", s_star - eps), ("after", s_star + eps)):
        model = sr0.robot(path_at(offset)(max(s, 0.0)))
        fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
        sides[label] = any(
            iks.configuration_distance(q_star, q, circular) < radius for q in fiber.solutions
        )
    if sides["before"] and not sides["after"]:
        return "death"
    if sides["after"] and not sides["before"]:
        return "birth"
    return "invisible"


def main() -> int:
    """Classify the folds, predict, measure, and tally."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--window", type=float, default=0.012)
    parser.add_argument("--eps", type=float, default=5e-3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.seed)
    agreed = disagreed = groups = events_total = folds_total = 0
    for pose in range(args.poses):
        pose_rng = np.random.default_rng(pose)
        q_true = pose_rng.uniform(-2.0, 2.0, sr5_model.num_dof)
        target = sr5_model.fk(q_true)
        seeds = [s.q for s in sr0.solve(sr0_model, target)]
        folds = folds_of(offset, target, seeds, near=args.near, iterations=args.iterations)
        folds_total += len(folds)
        if not folds:
            print(f"pose {pose}: no folds on the tracked branches (no events)")
            continue
        groups_list: list[list[tuple[float, np.ndarray]]] = []
        for fold in folds:
            if groups_list and fold[0] - groups_list[-1][-1][0] <= args.window:
                groups_list[-1].append(fold)
            else:
                groups_list.append([fold])
        current = seeds
        s_current = 0.0
        for group in groups_list:
            s_before = max(group[0][0] - args.eps, 0.0)
            s_after = min(group[-1][0] + args.eps, 1.0)
            kinds = [
                classify_fold(offset, target, s_star, q_star, circular, eps=args.eps,
                              seeds=args.seeds, rng=rng)
                for s_star, q_star in group
            ]
            events = [kind for kind in kinds if kind != "invisible"]
            predicted = 2 * sum(1 if kind == "birth" else -1 for kind in events)
            survivors, died = carry(offset, target, current, s_current, s_before, args.sigma_floor)
            born = births_at(offset, target, s_before, survivors, circular, seeds=args.seeds, rng=rng)
            before = survivors + born
            arrived, deaths = carry(offset, target, before, s_before, s_after, args.sigma_floor)
            added = births_at(offset, target, s_after, arrived, circular, seeds=args.seeds, rng=rng)
            after = arrived + added
            delta = len(after) - len(before)
            verdict = "agrees" if delta == predicted else "DISAGREES"
            agreed += int(verdict == "agrees")
            disagreed += int(verdict == "DISAGREES")
            groups += 1
            events_total += len(events)
            print(
                f"  pose {pose} window [{s_before:.3f}, {s_after:.3f}]: {len(group)} fold(s) "
                f"{kinds} -> events {len(events)}, predicted {predicted:+d} | measured "
                f"{len(before)} -> {len(after)} (delta {delta:+d}, {deaths} died, {len(added)} born) "
                f"{verdict}"
            )
            current = after
            s_current = s_after
    print(f"\ntotals: {folds_total} folds, {events_total} real events, {groups} windows | "
          f"law agrees in {agreed}, disagrees in {disagreed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
