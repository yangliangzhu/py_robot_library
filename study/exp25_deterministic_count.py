#!/usr/bin/env python3
"""exp25: counting the fibre deterministically, and testing the two-per-fold law with it

exp24 could not resolve the law because a census-based count carries a +-2 spread, the same size as
the signal.  The deterministic count splits the fibre into two parts that are measured differently:

* **survivors** -- carry every solution of the previous ``s`` forward with the predictor-corrector;
  what arrives is exact (continuity, no sampling);
* **births** -- solve the new arm from scratch and keep the solutions that match no survivor; these
  are the configurations that appeared inside the interval.

Then ``fibre(s_to) = survivors + births`` and ``fibre(s_to) - fibre(s_from)`` is a deterministic
difference up to the birth detection, which the fold structure predicts: ``+-2`` per fold crossed.

Run::

    python3 -m study.exp25_deterministic_count --pose-seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def path_between(offset: float, s0: float, s1: float):
    """The path restricted to ``[s0, s1]``, reparameterised to ``[0, 1]``."""
    full = path_at(offset)

    def make(t: float) -> list[np.ndarray]:
        return full(s0 + t * (s1 - s0))

    return make


def carry(offset: float, target, solutions: list[np.ndarray], s0: float, s1: float,
          sigma_floor: float) -> tuple[list[np.ndarray], int]:
    """Track every solution from ``s0`` to ``s1``; returns the survivors and how many died."""
    path = path_between(offset, s0, s1)
    survivors = []
    for q in solutions:
        track = homotopy.track(path, target, q, sigma_floor=sigma_floor)
        if track.finished:
            survivors.append(track.q)
    return survivors, len(solutions) - len(survivors)


def births_at(offset: float, target, s: float, survivors: list[np.ndarray], circular, *,
              seeds: int, rng, match: float = 1e-3) -> list[np.ndarray]:
    """Solutions of the arm at ``s`` that are not continuations of the survivors."""
    model = sr0.robot(path_at(offset)(s))
    fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
    return [
        q for q in fiber.solutions
        if not any(iks.configuration_distance(q, other, circular) < match for other in survivors)
    ]


def main() -> int:
    """Count the fibre deterministically across the fold groups of one pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--census-seeds", type=int, default=400)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=13)
    #: the fold groups measured in exp23/exp24: (before, after, folds in between)
    parser.add_argument("--windows", type=float, nargs="+",
                        default=[0.650, 0.662, 4, 0.710, 0.725, 1, 0.792, 0.806, 1, 0.950, 0.975, 2])
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.seed)
    pose_rng = np.random.default_rng(args.pose_seed)
    q_true = pose_rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)

    current = [s.q for s in sr0.solve(sr0_model, target)]
    s_current = 0.0
    print(f"pose seed {args.pose_seed}: start with {len(current)} solutions at s = 0 (SR0's fibre)")
    values = args.windows
    for index in range(0, len(values), 3):
        s_before, s_after, folds = values[index], values[index + 1], int(values[index + 2])
        at_before, deaths_before = carry(offset, target, current, s_current, s_before,
                                         args.sigma_floor)
        born_before = births_at(offset, target, s_before, at_before, circular,
                                seeds=args.census_seeds, rng=rng)
        count_before = len(at_before) + len(born_before)
        at_after, deaths_after = carry(offset, target, at_before, s_before, s_after,
                                       args.sigma_floor)
        born_after = births_at(offset, target, s_after, at_after, circular,
                               seeds=args.census_seeds, rng=rng)
        count_after = len(at_after) + len(born_after)
        delta = count_after - count_before
        print(
            f"  window [{s_before:.3f}, {s_after:.3f}] with {folds} fold(s): "
            f"fibre {count_before} -> {count_after} (delta {delta:+d}, predicted "
            f"{2 * folds:+d} in magnitude) | survivors {len(at_after)}/{len(at_before)} carried, "
            f"{deaths_after} died, {len(born_after)} born"
        )
        current = at_after + born_after
        s_current = s_after
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
